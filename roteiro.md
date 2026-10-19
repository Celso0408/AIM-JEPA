# Roteiro das mudanças de desempenho — branch `thais`

Documento de trabalho para a equipe do modelo (Celso, Kaike, Rafael) e para a DreamSquad.
Data: 19/10/2026. Base: `psi_JEPA_project`, commit `bc6f6a6`, revisão de código
`2026-09-25-expanded-domain-direct-supervision`.

**Nenhuma branch existente foi alterada.** Esta branch sai de `psi_JEPA_project` e contém
duas fases de mudança, descritas abaixo na ordem em que foram feitas.

---

## Por que mexer no carregamento, e não na máquina

O diagnóstico que originou este trabalho: o treino usa **um único núcleo** para preparar os
lotes, e refaz a cada época um trabalho que não muda. Isso tem três consequências práticas:

- a `g6.16xlarge` dobra as vCPUs, mas o código usa uma só;
- o `p4d` com 8 A100 não ajuda, porque não há treino distribuído implementado;
- o pico de VRAM medido é de 2,03 GB numa placa de 24 GB — não falta memória de GPU.

As mudanças abaixo atacam exatamente esse ponto, e nenhuma delas depende de hardware novo.

---

## Fase 1 — `s02_data.py` e `s05_training.py`

### 1. Arestas do grafo vetorizadas (`build_graph_edges`)

**O que era.** Um laço Python por aresta, com indexação escalar de tensores. Para a malha
de 257 pontos do STANDARD são cerca de 2.700 arestas dirigidas por grafo, reconstruídas a
cada época, para cada amostra, em treino **e** em validação.

**O que passou a ser.** A topologia (`edge_index`) depende apenas do número de pontos e das
dilatações, nunca dos valores — passa a ser calculada uma vez por tamanho de malha e
guardada em `EDGE_TOPOLOGY_CACHE`. Os atributos (`edge_attr`) passam a ser calculados com
numpy vetorizado.

**A ordem das arestas foi preservada exatamente**, porque a troca de mensagens soma sobre as
arestas de entrada e a ordem da soma afeta o resultado em ponto flutuante.

**Evidência de consistência:** `bench/bench_build_graph_edges.py` compara as duas
implementações com `torch.equal` nos dois tensores devolvidos.

| Cobertura | Casos |
|---|---|
| Geometrias | irregular (tipo FEM), uniforme, radial |
| Tamanhos de malha | n = 1, 2, 3, 17, 65, 129, 257, 513 |
| Lotes | 1 a 3 amostras, incluindo lotes de comprimentos mistos |
| **Resultado** | **32 de 32 exatos, diferença máxima 0,0** |

**Evidência de ganho:** mesmo script, melhor de N repetições, um núcleo.

| Modo | n | lote | arestas | antes | depois | ganho |
|---|---|---|---|---|---|---|
| DEV | 129 | 8 | 10.856 | 1.082 ms | 11,2 ms | 96× |
| STANDARD | 257 | 8 | 22.120 | 2.041 ms | 10,7 ms | 191× |
| FULL | 513 | 8 | 44.648 | 4.707 ms | 14,8 ms | 317× |

Extrapolando para o DEVELOPMENT, que é o único modo com execução registrada nesta revisão
do código: 160,6 ms por amostra, ou cerca de 139 s por época só para construir arestas,
contra uma época real medida de 263,3 s. Depois da mudança, 0,63 s.

A medição foi feita numa CPU local, provavelmente mais lenta que a da nuvem — o tempo
absoluto é um limite superior, mas o ganho relativo independe da máquina.

### 2. Validação em uma passada (`validation_summary`)

**O que era.** Uma passada global sobre a validação e, em seguida, para cada família, a
construção de um `DataLoader` novo e uma passada inteira de novo — 27 vezes.

**O que passou a ser.** Uma passada só, com lotes que nunca misturam famílias, acumulando
as métricas global e por família ao mesmo tempo.

**Evidência de consistência:** `bench/test_agregacao_validacao.py` reproduz a aritmética de
agregação dos dois caminhos.

| Métrica | Equivalência | Erro medido |
|---|---|---|
| Por família | **Exata** | 0 — são os mesmos lotes: mesmas linhas, mesma ordem, mesmo tamanho |
| Global | Até arredondamento | ~1e-16 na agregação, em 6 cenários |

**Ponto que precisa de decisão de vocês.** Como os lotes deixam de misturar famílias, a
ordem das somas muda e a métrica global muda no último dígito. Há ainda o arredondamento em
float32 de cada termo, que depende da composição do lote. Como o `robust_validation_score`
escolhe o checkpoint, uma disputa muito apertada entre duas épocas poderia resolver-se para
o outro lado. É improvável e é arredondamento, não erro — mas é mudança de comportamento e
precisa ser dita antes.

Se preferirem preservar o número global bit a bit, dá para manter a passada global como
estava e aplicar só o item 3 abaixo: fica a maior parte do ganho de preparação e nenhum
ganho de GPU.

### 3. Lotes de validação em cache

Os registros de validação e teste são imutáveis por construção — o `HamiltonianDataset` é
criado com `augment=False` sempre que `shuffle=False`. Os lotes colacionados passam a ser
construídos uma vez e reutilizados em todas as épocas.

**Equivalência: exata**, são os mesmos tensores.
**Custo:** memória de host, na casa de 150 a 250 MB no STANDARD. Desligue com
`PSI_JEPA_CACHE_VALIDATION_BATCHES=0`.

### 4. Cópia rasa no `__getitem__`

O `copy.deepcopy` por amostra de treino saiu. As augmentações acima já devolvem dicionários
novos, e a colagem só lê os arrays: `torch.as_tensor` escreve dentro de um tensor
pré-alocado, o que copia. A cópia rasa basta para anexar `_request_reflection_pair` sem
tocar no registro guardado.

**Equivalência: exata**, verificada por leitura do caminho de dados. O DEVELOPMENT pareado
confirma na prática.

---

## Fase 2 — `s02_data.py`

### 5. Limpeza do cache de reamostragem (`set_epoch`) — **mudança sugerida**

Esta é a mudança que vale discussão, e não tem a ver com velocidade.

**O problema.** O `resample_cache` é indexado por `(group_id, epoch)`. No
`split_manifest.json` do STANDARD são 2.430 linhas e 2.430 `group_id` distintos: a chave é
única por registro. Como cada índice é visitado uma vez por época, cada entrada é escrita e
lida uma única vez, dentro da mesma época, e nunca mais pode ser alcançada — a época faz
parte da chave. Nada limpa o dicionário.

O resultado é crescimento sem limite, proporcional a épocas × amostras reamostradas:

| Modo | Por registro | Reamostrados/época | Épocas | Acúmulo ao fim |
|---|---|---|---|---|
| DEVELOPMENT | 26 KB | 89 | 44 | 0,11 GB |
| STANDARD | 53 KB | 364 | 165 | **3,3 GB** |
| FULL | 105 KB | 466 | 180 | **9,1 GB** |

A conta assume `resampling_probability = 0,15`, arrays em float64 e os campos do esquema
(`x`, `quadrature_weights`, `V_raw`, `grid_spacing`, `psi`, `rho` e as duas máscaras). Os
valores são ordem de grandeza. **A existência do vazamento não é estimativa**: é leitura
direta do código.

**A correção** é limpar o cache quando a época muda. Nenhum valor amostrado se altera,
porque as entradas descartadas pertencem a épocas que já passaram.

**Por que isso importa para o dimensionamento.** O pedido de `g6.16xlarge` dobra a RAM de
host, de 120 para 254 GB. Vale checar quanto dessa necessidade é este vazamento. A medição
responde sozinha: se a RAM crescer de forma monotônica ao longo das épocas, é ele.

### 6. `num_workers` no loader de treino

A preparação dos lotes de treino passa a rodar em processos paralelos, com três salvaguardas:

- **Só com `fork`.** Os estágios são executados com `exec()` dentro do namespace do
  `main.py`, então `HamiltonianDataset` fica num módulo que um processo filho não consegue
  importar. Com `fork` (Linux) o filho herda os objetos e nada é serializado; com `spawn`
  (Windows, macOS) o worker falharia ao reconstruir o dataset. A contagem é forçada a 0 fora
  do `fork`, com aviso se alguém tentar forçar pela variável de ambiente.
- **`persistent_workers=False`,** porque o `s05` chama `set_epoch` entre épocas. Com workers
  persistentes, a cópia do dataset no worker ficaria presa na época 0 e a augmentação
  congelaria **sem levantar erro**.
- **Só no loader de treino.** Validação e teste já têm os lotes em cache pelo item 3.

Padrão: `min(8, núcleos − 2)`. Ajuste com `PSI_JEPA_DATALOADER_WORKERS`; `0` volta ao
comportamento atual.

**Equivalência: exata.** A aleatoriedade do `__getitem__` vem de um hash de
`(seed, epoch, group_id)`, não do RNG global, então não depende do worker nem da ordem em
que os índices são processados. Verifiquei que não há uso de RNG global no caminho do
dataset.

**Uma perda a registrar:** `resample_failures` passa a viver dentro de cada worker e some no
fim da época. Hoje essa lista é escrita e **nunca lida** por nenhum estágio — conferi em
todos os arquivos. Não é regressão, mas se um dia for preciso auditar falhas de
reamostragem, elas terão de voltar pelo lote.

---

## Como verificar

Os dois scripts são independentes do repositório e do modelo. Rodam em CPU, em segundos:

```bash
python bench/bench_build_graph_edges.py      # arestas: equivalência + benchmark
python bench/test_agregacao_validacao.py     # agregação da validação
```

Os dois terminam com `RESULTADO: equivalencia ... confirmada` e código de saída 0. Cada um
carrega a sua própria cópia das duas implementações, então continuam válidos como teste de
regressão independentemente do estado do repositório.

Na sequência, o critério de aceite do pipeline:

```bash
python -u main.py --mode QUICK                 # os 68 testes em PASS
python -u main.py --mode DEVELOPMENT --seed 7  # comparar com a execução atual
```

No DEVELOPMENT pareado, comparar `run_card.json`, campo `scientific_acceptance`, e as
colunas de `training_history.csv`. As métricas por família devem bater exatamente; a global,
até arredondamento.

## Consequência operacional

Alterar os estágios muda o `code_hash` e, com ele, o nome do diretório de saída. **Uma
execução em andamento não retoma depois destas mudanças.** Elas entram entre ciclos.

## Como reverter

Cada fase é um commit separado. `git revert` do commit da fase 2 desfaz o `num_workers` e a
limpeza do cache sem tocar na fase 1.

---

## O que não está aqui, e por quê

| Item | Por que ficou de fora |
|---|---|
| Microlote maior que 2 | Depende de medir o pico de VRAM com o treino rodando |
| TF32 e precisão mista | Hoje `allow_tf32` está desligado e `float32_matmul_precision` é `highest`. É a configuração mais conservadora e também a mais lenta. Mexer nisso exige comparação pareada das portas numéricas, porque há limiares de 1e-4 e 1e-7 em jogo |
| Checkpoint sub-época | Destravaria instâncias spot, com 45% a 78% de desconto. A retomada já existe e é atômica; falta só a granularidade, hoje de uma época — 1,37 h no FULL |
| Treino distribuído | Não se justifica para um modelo de ~10⁶ parâmetros. O ganho medido pela própria equipe com 8 A100 foi de cerca de 2×, eficiência de 25% |

## A medição que continua pendente

Nada aqui substitui a medição de uma hora na máquina de treino, pedida desde 29/09 e
detalhada na seção 1.3 do documento da DreamSquad: utilização da GPU com `nvidia-smi`, pico
de VRAM, pico de RAM ao longo do ciclo, repartição do tempo com o profiler e custo isolado
do pré-processamento. Ela diz qual fração da época cada item acima realmente devolve — e,
no caso do item 5, se o vazamento explica parte do pedido de RAM.

Um dado que reforça o pedido: a execução STANDARD de 49,7 h registrada em `outputs/` vem da
revisão `2026-07-20`, enquanto a árvore de trabalho está em `2026-09-25`. **Não existe hoje
um número de referência para o código que está rodando.**
