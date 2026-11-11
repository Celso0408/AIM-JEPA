# Do New-Schrodinger_JEPA à V12: o que mudou em cada versão

**Pergunta:** o que mudou desde o treino em `New-Schrodinger_JEPA` até a V12, passando por V08, V10 e V11? E o que explica o resultado da V12 STANDARD?

**Resposta curta:**

- A linha que chega à V12 tem **quatro mudanças algorítmicas reais**, todas feitas dentro da V08:
  1. janela de operador sem uso do alvo (v6 com janela);
  2. seletor `physics_selected`, checkpoint robusto e pesos rebalanceados (v7);
  3. domínio ampliado em 60%, gate topológico e supervisão direta permanente (v8);
  4. o protocolo opcional com Kratzer fora do treino.
- **V10 e V11 foram ramos experimentais.** Nenhuma das mudanças testadas neles entrou na V12. Nenhuma superou o controle V08 no DEVELOPMENT pareado.
- **A V12 é o código v8 mais instrumentação observacional.** O algoritmo de treino é o mesmo do FULL v8 e do controle V08.
- **A V12 FULL terminou em 31/10 com 0,979 na rota direta**, praticamente igual à V12 STANDARD (0,977) e muito acima da FULL v8 (0,897), com o mesmo código, configuração e dados. A queda da FULL v8 **não** vinha do preset FULL: foi uma instabilidade do caminho de execução daquela run (seção 4.3).
- A V12 STANDARD (0,977 direta) supera o FULL do New-Schrodinger (0,919). O ganho veio sobretudo da **janela de operador** introduzida na V08 v6. A melhor run continua sendo a **STANDARD v6 da V08** (0,986), que já tinha essa janela e ainda não tinha o seletor. Comparações entre presets diferentes não são pareadas (seção 4).

Todas as runs são de seed única. Ver as ressalvas na seção 7.

---

## 1. Mapa das versões

Hashes recomputados com o mesmo algoritmo de `s00_bootstrap.py` (`SOURCE_SHA256`, 12 primeiros caracteres) e conferidos com os `run_card.json`.

| Etapa | Pasta | Data | `source_sha256` | Arquitetura | Mudança principal | Herdada pela V12? |
|---|---|---|---|---|---|---|
| 0 | `New-Schrodinger_JEPA` | ago/2026 | `ec4611f11ae8` (FULL) | v6 | Ponto de partida: decoder `topology_phase`, domínio original sem janela | base |
| 1 | `psi_Jepa_V08` (v6) | 22–25/09 | `8ff50bf69e81` | v6 | Janela de operador de 70% sem uso do alvo; entrada `min(V, Vcap)` | **sim** |
| 2 | `psi_Jepa_V08` (v7) | 25/09 | `b1a5632a64e4` | v7 | `physics_selected` publicado; checkpoint robusto por grupo; pesos rebalanceados | **sim** |
| 3 | `psi_Jepa_V08` (v8) | 25/09 | `240831cc275f` | v8 | Janela +60%; auditoria de cauda a 99%; seletor com gate topológico; supervisão direta permanente | **sim** |
| 3b | `psi_Jepa_V08` (v8 + opção) | 09/10 | `f39e4db6dc7f` | v8 | Opção `heldout_kratzer` (Kratzer passa para OOD) | **sim** (opção) |
| 4 | `psi_Jepa_V10` | 08–09/10 | vários | v8 | Perdas sobre a rota direta, posição de nós, massa de lobos prevista, flag "nenhuma rota válida", score por célula família × estado | não |
| 5 | `V11` | 09–13/10 | `5447e68da057` → `7a352d8ba6a5` | v8 | Controle v8; perda relativa de lobos; currículo gradual 6→8→10→11 | não |
| 6 | `V12` | 13/10 → | `900470072850` / `3e24fb0b898f` | v12 | Código v8 + `s08_latent_diagnostics.py` e checkpoints de diagnóstico | — |

Árvores de código usadas na reconstrução:

- **v6 sem janela:** `New-Schrodinger_JEPA/` atual, hash `1fde5d21bd83`. Os arquivos `s03`–`s07` são de 15–16/08 e idênticos aos da V08 v6. Os arquivos `s00`–`s02` foram editados em 09/10 para receber a opção `heldout_kratzer`. Por isso o código exato do FULL de agosto (`ec4611f1`) não existe mais em disco. A diferença se limita a essa opção e à ordem de famílias.
- **v6 com janela:** `psi_Jepa_V08/backup/`, cujo hash coincide com o da STANDARD v6 `cf43`.
- **v8:** `psi_Jepa_V08.zip`, igual ao commit `HEAD` do git da `psi_Jepa_V10`.
- **V10 e V11:** o diff de cada pasta contra a v8.

A pasta `psi_Jepa_V08/` atual tem hash `67e8cd1409d6`. A única diferença em relação a `f39e4db6dc7f` é `num_workers=2` no DataLoader de `s02_data.py`, editado em 22/10. Essa mudança é operacional, nenhuma run a usou, e a V12 mantém `num_workers=0`.

## 2. Etapa por etapa

### Etapa 0 — `New-Schrodinger_JEPA` (v6, domínio sem janela)

O código é uma exportação modular do notebook original. A arquitetura é `psi-true-jepa-1d-v6-topology-phase-operator-energy`:

- backbone GNN-KAN;
- decoder `topology_phase` com refinamento recorrente;
- **saída publicada = rota direta** (`direct_topology`).

O domínio de cada Hamiltoniano é o de geração. As famílias radiais, por exemplo, vão de `r = 0` até `max(rmax, 90)`, sem corte posterior.

| Run | Diretório | Preset | Resultado ID (direta) |
|---|---|---|---|
| FULL v6 | `outputs/true_jepa_full_221e21da_a46c65a5` (25/08) | FULL | 0,919 |
| DEV v6, Kratzer fora | `outputs/true_jepa_development_40f43551_3de7732b` (09/10) | DEVELOPMENT | 0,472 |

A FULL foi analisada em `problems/prob2.txt`. A regressão foi ampla: Morse 0,78, Pöschl–Teller H 0,82, Kratzer 0,84. Os nós persistentes ficaram em apenas 0,65. Em Kratzer, o "raw node accuracy" valia 1,0, mas os nós persistentes valiam 0,09: o decoder impõe o número formal de cruzamentos sem posicioná-los corretamente.

O DEV de 09/10 foi lançado junto com os controles V08 e V10 sem Kratzer. Sem a janela, o regime DEVELOPMENT praticamente não aprende (0,47).

### Etapa 1 — V08 v6: janela de operador sem uso do alvo

A diferença de código para o New-Schrodinger está concentrada em `s01_physics.py` e `s02_data.py`. `s03` e `s05`–`s07` são idênticos.

- **`potential_only_operator_window`:** uma estimativa Weyl/WKB, calculada só a partir de `V(x)`, define a energia do estado mais alto. O domínio artificial é encolhido para que esse estado ocupe cerca de 70% da faixa classicamente acessível. Intervalos físicos (finitos ou singulares) não mudam.
- **Entrada limitada:** a rede passa a receber `min(V, Vcap)` normalizado, em vez do `V` bruto. O Hamiltoniano físico continua exato.
- **Auditoria:** novos campos `operator_window_audit.csv` e massa visível do estado mais alto a 95%.
- Versões de cache numérico e de dataset atualizadas, o que regenera os dados.
- Operacional: `checkpointing.py` passou a repetir a troca atômica de arquivos quando ela falha por bloqueio no Windows.

| Run | Diretório | Resultado ID (direta) |
|---|---|---|
| STANDARD v6 | `psi_Jepa_V08/outputs/true_jepa_standard_cf43e033_b7c28c2b` | **0,986**, a melhor run de toda a linha |
| DEV v6 | `psi_Jepa_V08/outputs/true_jepa_development_87cb89e4_bc5e6be0` | 0,873 |

### Etapa 2 — V08 v7: `physics_selected`

Descrito em `V12/tests/last_changes.md`:

- **Seletor `physics_selected`.** Escolhe uma rota por Hamiltoniano entre a direta, o Ritz inicial e o Ritz refinado, com um score que não usa o alvo (resíduo, Gram e topologia).
- **Checkpoint robusto.** O score combina média global, pior quartil de grupos e pior grupo.
- **Pesos aumentados:** resíduo, resíduo profundo, contração, ortogonalidade e supervisão do subespaço Ritz nos estágios de 11 estados.

DEV v7 (`dad1c612`): a fidelidade publicada sobe para 0,928, mas a **direta cai para 0,305**, e os nós persistentes da direta para 0,38 (`problems/prob5.txt`). O seletor escondia o colapso do decoder.

### Etapa 3 — V08 v8: domínio ampliado e supervisão direta

Segunda parte de `last_changes.md`:

- **Janela ampliada em 60%.** Nas linhas truncadas, 30% de cada lado. Em meia-linha e radial, só a borda direita cresce. A malha cresce na mesma proporção, preservando o passo espacial.
- **Auditoria de cauda mais rígida:** de 95% para 99%, com limite de massa na casca externa reduzido de 5% para 1%.
- **Gate topológico no seletor.** Uma rota que falha na topologia de nós recebe score infinito. O peso de Gram cai de 0,25 para 0,05.
- **Supervisão direta permanente** (onda, infidelidade, densidade, H1 e projetor), mesmo quando outra rota é publicada.

| Run | Diretório | Direta | Publicada |
|---|---|---|---|
| DEV v8 | `true_jepa_development_2d4f9847_0d182910` | 0,880 | 0,881 |
| FULL v8 | `true_jepa_full_593bf82a_0d182910` | 0,897 | 0,903 |

A rota direta se recupera. A FULL v8 usa **o mesmo preset** da FULL do New-Schrodinger e fica abaixo dela na direta (0,897 contra 0,919) e nos nós persistentes (0,41 contra 0,65), embora tenha Ritz melhor (0,974 contra 0,963). A V12 FULL, com o mesmo código v8, mostrou depois que essa FULL v8 foi uma run instável (seção 4.3).

### Etapa 3b — Kratzer fora do treino (opção `heldout_kratzer`)

`PSI_JEPA_HELDOUT_KRATZER=1` remove Kratzer das famílias ID e o avalia como OOD. Isso muda a pergunta experimental, **não melhora Kratzer**. As médias ID dessas runs não são comparáveis com as runs que treinam com Kratzer.

Controle: `psi_Jepa_V08/outputs/true_jepa_development_9558d769_18adb670`, com 0,917 direta e 0,921 publicada. Kratzer OOD fica em 0,061.

### Etapa 4 — V10: ramo experimental, não herdado

Implementa parte das propostas de `problems/sol_v10.txt`. Diff contra a v8 (git `HEAD` da `psi_Jepa_V10`):

- As perdas principais (onda, densidade, resíduo, energia) passam a usar a **rota direta** `psi_direct`, e não a rota selecionada.
- Novas perdas `topology_node_position` (0,30) e `topology_predicted_lobe_mass` (0,20), aplicadas sobre os nós e lobos **previstos**.
- Flag explícita `physics_route_all_invalid`. Antes, quando todas as rotas falhavam, o `argmin` devolvia a rota 0 silenciosamente.
- Score de checkpoint com uma parcela de 20% para as piores células família × estado.
- Auditoria por família × estado a cada época e checkpoints de fronteira E6/E11.
- `direct_supervision_scale` e ramificação a partir do checkpoint E6.

Sequência de DEVELOPMENTs, todos com o mesmo split e normalização do DEV v8:

| Run | O que mudou | Direta | Publicada |
|---|---|---|---|
| `2d4f9847_36618359` | Tudo acima; os termos `direct_*` antigos ficaram com peso 0 | 0,863 | 0,864 |
| `2d4f9847_f8087c33` | Pesos `direct_*` restaurados ao valor da v8 | 0,888 | 0,889 |
| `2d4f9847_1b7ac958` | Só diagnósticos (fronteiras, auditoria) | 0,888 | 0,889 (métricas byte a byte iguais à `f808`) |
| `acad805b_f1bae9ca` | Ramo da E6 da `1b7a` com supervisão direta × 0,5 | 0,885 | 0,886 |
| `ccb511f6_a86e3923` | Kratzer fora | 0,902 | 0,903 (controle V08: 0,917 / 0,921) |

A melhor variante com Kratzer no treino (`f808`) ganha cerca de 0,8 ponto percentual sobre o DEV v8, mas perde nos nós persistentes (0,90 contra 0,93). No protocolo sem Kratzer, a V10 fica abaixo do controle V08. **Nada foi promovido.**

### Etapa 5 — V11: ramo experimental, não herdado

Resumo de `V11/tests/contex.txt`. Todas as runs deixam Kratzer fora e usam o mesmo split e normalização.

| Run | O que mudou | Direta | Publicada |
|---|---|---|---|
| `2b4c3084_3eed5340` | Controle: código v8 com `heldout_kratzer` | 0,916 | 0,919 |
| `64a59f66_21099521` | Perda **relativa** de massa dos lobos, com piso de 0,05 | 0,899 | 0,902 |
| `cd3bdb64_4ef18b99` | Perda relativa revertida; **currículo gradual** em E11 (8 → 10 → 11 estados) | 0,905 | 0,907 |

Detalhes das runs:

- A run `64a5` falhou em `s07` por um erro operacional na contagem de testes e não tem `run_card`. Mesmo assim, `half_harmonic` caiu para 0,54.
- O currículo gradual também não superou o controle.
- O código atual da V11 é o da `cd3b`.

A conclusão da V11 foi congelar o controle v8 e investigar o colapso latente por estado. Daí nasceu a V12.

### Etapa 6 — V12: controle v8 com instrumentação

V12 é uma cópia enxuta do controle V08, mantendo a opção `heldout_kratzer` (`V12/tests/V12_EXPERIMENT.md`). As mudanças em relação ao V08 são só observacionais:

- novo estágio `s08_latent_diagnostics.py`;
- snapshots `latent_checkpoint_*.pt` na transição 6→11;
- monitor de validação das famílias radiais nos estágios F e G;
- novos rótulos de versão.

Nenhuma delas altera perdas, otimização ou seleção de checkpoint. Prova direta: o DEV V12 (`49cb31c7`) reproduz **byte a byte** o `per_state_metrics.csv` do controle V11 `2b4c`.

| Run | Diretório | Preset | Kratzer | Direta | Publicada |
|---|---|---|---|---|---|
| DEV V12 | `V12/outputs/true_jepa_development_49cb31c7_b9825efe` | DEVELOPMENT | fora | 0,916 | 0,919 |
| **STANDARD V12** | `V12/outputs/true_jepa_standard_fbc1677c_aee2f53f` | STANDARD | **no treino** | 0,977 | 0,978 |
| FULL V12 | `V12/outputs/true_jepa_full_5ba6bc71_aee2f53f` | FULL | no treino | 0,979 | 0,979 |

## 3. Tabela consolidada (teste ID)

Média sobre pares Hamiltoniano × estado. "Energia" é a mediana por Hamiltoniano do erro absoluto. "Pior H" é a fidelidade direta do pior Hamiltoniano.

| Run | Código | Preset | Kratzer | # H ID | Direta | Ritz inicial | Publicada | H1 direta | Nós pers. | Energia publ. | Pior H | Estado 3 (direta) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NS FULL | v6 sem janela | FULL | ID | 783 | 0,919 | 0,963 | 0,919 | 0,225 | 0,653 | 0,082 | 0,702 | 0,919 |
| NS DEV | v6 sem janela | DEV | OOD | 130 | 0,472 | 0,660 | 0,472 | 0,768 | 0,880 | 0,171 | 0,150 | 0,373 |
| V08 DEV v6 | v6 janela | DEV | ID | 135 | 0,873 | 0,949 | 0,873 | 0,305 | 0,980 | 0,123 | 0,141 | 0,825 |
| **V08 STD v6** | v6 janela | STD | ID | 513 | **0,986** | **0,997** | **0,986** | **0,082** | **0,947** | 0,046 | **0,734** | **0,983** |
| V08 DEV v7 | v7 | DEV | ID | 135 | 0,305 | 0,928 | 0,928 | 0,955 | 0,384 | 0,019 | 0,096 | 0,334 |
| V08 DEV v8 | v8 | DEV | ID | 135 | 0,880 | 0,934 | 0,881 | 0,307 | 0,926 | 0,244 | 0,124 | 0,860 |
| V08 FULL v8 | v8 | FULL | ID | 783 | 0,897 | 0,974 | 0,903 | 0,291 | 0,409 | 0,097 | 0,661 | 0,811 |
| V08 DEV controle | v8 | DEV | OOD | 130 | 0,917 | 0,962 | 0,921 | 0,240 | 0,941 | 0,192 | 0,417 | 0,891 |
| V10 `3661` | V10 | DEV | ID | 135 | 0,863 | 0,937 | 0,864 | 0,334 | 0,935 | 0,250 | 0,107 | 0,829 |
| V10 `f808`/`1b7a` | V10 | DEV | ID | 135 | 0,888 | 0,936 | 0,889 | 0,285 | 0,902 | 0,308 | 0,383 | 0,875 |
| V10 `acad` (×0,5) | V10 | DEV | ID | 135 | 0,885 | 0,942 | 0,886 | 0,296 | 0,930 | 0,305 | 0,168 | 0,843 |
| V10 `ccb5` | V10 | DEV | OOD | 130 | 0,902 | 0,948 | 0,903 | 0,269 | 0,942 | 0,228 | 0,466 | 0,864 |
| V11 controle `2b4c` | v8 | DEV | OOD | 130 | 0,916 | 0,963 | 0,919 | 0,245 | 0,936 | 0,192 | 0,434 | 0,892 |
| V11 lobos `64a5` | V11 | DEV | OOD | 130 | 0,899 | 0,949 | 0,902 | 0,271 | 0,936 | 0,204 | 0,426 | 0,867 |
| V11 gradual `cd3b` | V11 | DEV | OOD | 130 | 0,905 | 0,953 | 0,907 | 0,261 | 0,933 | 0,186 | 0,422 | 0,891 |
| V12 DEV | v12 (= v8) | DEV | OOD | 130 | 0,916 | 0,963 | 0,919 | 0,245 | 0,936 | 0,192 | 0,434 | 0,892 |
| **V12 FULL** | v12 (= v8) | FULL | ID | 783 | **0,979** | 0,994 | 0,979 | 0,100 | 0,870 | **0,002** | 0,724 | 0,972 |
| **V12 STD** | v12 (= v8) | STD | ID | 513 | 0,977 | 0,995 | 0,978 | 0,103 | 0,880 | **0,004** | 0,670 | 0,968 |

Comparações válidas como pares (mesmo split e normalização, ou mesmo preset):

- **Com Kratzer, DEVELOPMENT, v6 × v7 × v8:** dados idênticos entre v6 e v7 (`problems/prob5.txt`). A v8 regenerou os dados ao ampliar a janela.
- **V10 × DEV v8:** mesmo split e normalização.
- **Sem Kratzer:** controle V08, V10 `ccb5`, V11 (três runs) e DEV V12 compartilham o split.
- **NS FULL × V08 FULL v8:** mesmo preset, mas dados diferentes, porque a janela muda domínio e malha.

## 4. O que explica a V12 STANDARD

### 4.1 Em relação ao New-Schrodinger: a janela de operador

A maior mudança de qualidade da linha acontece na etapa 1. A janela de 70% com entrada limitada (`Vcap`) leva o regime DEVELOPMENT de inviável (0,47 no New-Schrodinger) a 0,87. Também produz a melhor STANDARD (0,986). A V12 herda essa janela, ampliada em 60% na v8.

O ganho mais visível da V12 STANDARD sobre a FULL do New-Schrodinger está nas famílias que lá falharam:

- `morse` vai de 0,78 para valores acima de 0,99;
- `poschl_h` vai de 0,82 para valores acima de 0,99;
- `kratzer` vai de 0,84 para 0,93;
- os nós persistentes vão de 0,65 para 0,88.

Ressalva: os presets são diferentes (FULL × STANDARD), então esta não é uma comparação pareada.

### 4.2 FULL v8 × STANDARD V12: a diferença não era o preset

A FULL v8 e a STANDARD V12 usam o mesmo código e diferem no preset:

| Parâmetro | FULL (v8 e New-Schrodinger) | STANDARD V12 |
|---|---|---|
| Hamiltonianos por família | 192 | 128 |
| Nós de referência / modelo / validação | 2049 / 513 / 1025 | 1025 / 257 / 513 |
| Largura oculta (`hidden`) | 160 | 128 |
| Épocas AE / JEPA | 60 / 30 | 32 / 24 |
| Passos de refinamento (treino / inferência) | 5 / 10 | 3 / 5 |

Fidelidade de validação no fim de cada estágio, e estado 3 entre parênteses:

| Estágio | NS FULL (v6) | FULL v8 | **FULL V12** | STANDARD V12 |
|---|---|---|---|---|
| `E_subspace_3` | 0,948 (0,15) | 0,963 (0,11) | 0,961 (0,08) | 0,957 (0,58) |
| `E_subspace_6` | 0,894 (0,91) | **0,880 (0,76)** | **0,966 (0,96)** | **0,965 (0,95)** |
| `E_supervised_11` | 0,910 | 0,894 | 0,976 | 0,977 |
| `G_joint_finetune` | 0,916 | 0,898 | 0,979 | 0,979 |

Estados 3 / 4 / 5 na validação, na última época do `E_subspace_6`:

| NS FULL (v6) | FULL v8 | FULL V12 | STANDARD V12 |
|---|---|---|---|
| 0,91 / 0,79 / 0,80 | 0,76 / 0,79 / 0,81 | 0,96 / 0,96 / 0,97 | 0,96 / 0,96 / 0,97 |

### 4.3 Resultado da V12 FULL: a hipótese do preset foi refutada

A versão anterior deste documento atribuía a queda da FULL v8 ao preset FULL (alcance do grafo com dilatações fixas). **A V12 FULL refuta essa hipótese.**

- A V12 FULL e a FULL v8 têm **a mesma configuração** (o diff de `config.json` só mostra o rótulo de arquitetura e `heldout_kratzer=False`) e **os mesmos dados**: os hashes de dataset, split e normalização são iguais. O algoritmo de treino também é o mesmo; a V12 só acrescenta instrumentação.
- Mesmo assim, a V12 FULL **não** teve a queda no `E_subspace_6`: estados 3–5 terminam em 0,96 / 0,96 / 0,97, igual à STANDARD V12, contra 0,76 / 0,79 / 0,81 na FULL v8.
- No teste ID, a V12 FULL fica em 0,979 (direta), contra 0,897 da FULL v8. O estado 3 vai de 0,81 para 0,97 e os nós persistentes de 0,41 para 0,87.

**O que difere entre as duas runs é o caminho de execução:**

| | FULL v8 | FULL V12 |
|---|---|---|
| Autoencoder de soluções | treinado com microbatch 2 / acumulação 8 (tentativa de 26/09, que depois deu OOM) | microbatch 1 / acumulação 16 |
| Treino JEPA e currículo | microbatch 1; interrompido e **retomado** em `D_jepa_latent`, época 10 | microbatch 1; contínuo, sem retomada |

Com seed 7 e algoritmos determinísticos, essas diferenças bastam para gerar pesos diferentes. O autoencoder, em particular, define os latentes-alvo de todo o treino seguinte. Não dá para saber qual das duas diferenças importou, mas a conclusão principal não depende disso: **o mesmo código, com os mesmos dados e o mesmo preset, chega a 0,90 ou a 0,98 conforme o caminho de execução.** A queda da FULL v8 foi uma instabilidade de otimização, e não um efeito do preset.

Consequências:

- A comparação FULL v8 × STANDARD V12 (seção 4.2) não mede efeito de preset; mede variação entre runs.
- A **variância entre runs** do código v8 é muito maior do que se supunha. Diferenças de alguns pontos entre runs de seed única (inclusive V10 e V11 contra o controle) podem estar dentro dessa variação.
- A FULL do New-Schrodinger também ficou presa nos estados 4–5. Com uma única run, isso pode ser o mesmo tipo de instabilidade, e não algo próprio da v6.

**Diagnóstico latente (s08).** A V12 FULL tem latentes previstos bem mais próximos dos latentes-alvo que a STANDARD (cosseno 0,57 contra 0,29 no fim do E11; L2 relativo 1,23 contra 3,91). A fidelidade de saída, porém, é a mesma (0,976 nos dois checkpoints selecionados). Ou seja, alinhar melhor o latente não se traduziu em ganho de fidelidade.

### 4.4 A rota publicada `physics_selected` (v7/v8)

Ela não torna a rede melhor; muda **qual saída é publicada**.

| Rota (ID, STANDARD) | Fidelidade v6 | Fidelidade V12 | Energia v6 | Energia V12 |
|---|---|---|---|---|
| Direta (rede) | **0,986** | 0,977 | **0,046** | 0,101 |
| Ritz inicial | 0,997 | 0,995 | 0,0019 | 0,0022 |
| Publicada | 0,986 (direta) | 0,978 (`physics_selected`) | 0,046 | **0,004** |

- O ganho de energia da V12 vem de o seletor publicar o Ritz inicial em 51% dos Hamiltonianos. O Ritz da v6 já era tão bom quanto o da V12; a v6 apenas não o publicava.
- O filtro de topologia descarta o Ritz inicial (fidelidade 0,995) em 45% dos Hamiltonianos. Em 142 de 513 casos, a rota direta é publicada mesmo sendo mais de 1 ponto percentual pior.

### 4.5 V10 e V11 não contribuem

Nenhuma mudança desses ramos está no código da V12. A V12 STANDARD é, portanto, o primeiro teste em escala STANDARD do código **v8 puro**.

## 5. Comparação honesta: V12 STANDARD × V08 STANDARD v6

| Métrica (ID, direta) | STANDARD v6 | STANDARD V12 | Melhor |
|---|---|---|---|
| Fidelidade direta | 0,986 | 0,977 | v6 |
| Fidelidade publicada | 0,986 | 0,978 | v6 |
| H1 relativo | 0,082 | 0,103 | v6 |
| Nós persistentes corretos | 0,947 | 0,880 | v6 |
| Pior Hamiltoniano | 0,734 | 0,670 | v6 |
| `radial_oscillator` | 0,999 | 0,854 | v6 |
| `half_harmonic` | 1,000 | 0,910 | v6 |
| `fourier_random` / `gaussian_random_field` | 0,958 / 0,945 | 0,969 / 0,950 | V12 |
| Energia publicada (mediana) | 0,046 | 0,004 | V12 (efeito do seletor) |
| OOD não visto (fidelidade publicada) | 0,772 | 0,697 | v6 |
| Critérios reprovados no `run_card` | 7 | 8 | v6 |

Entre a v6 e a v8 mudaram juntos o seletor, o checkpoint robusto, os pesos, a janela +60% (que regenerou os dados), o gate topológico e a supervisão direta. Como nunca houve ablação pareada, **a regressão de forma de onda não pode ser atribuída a uma mudança específica.**

## 6. Resumo da evolução

```
New-Schrodinger v6 (sem janela)       FULL 0,919 · DEV 0,47
   └─ V08 v6: janela de operador 70%  STD 0,986 · DEV 0,873          ← maior salto
       └─ V08 v7: physics_selected    DEV direta 0,305 / publicada 0,928  (colapso mascarado)
           └─ V08 v8: janela +60%, gate topológico, supervisão direta
               │                      DEV 0,880 · FULL 0,897
               ├─ V10 (ramo)          DEV 0,863–0,888; sem Kratzer 0,902  → descartado
               ├─ V11 (ramo)          lobos rel. 0,899; gradual 0,905 (controle 0,916) → descartado
               └─ V12 = v8 + diagnóstico  DEV 0,916 (= controle) · STD 0,977 · FULL 0,979
```

## 7. Ressalvas

- **Seed única em todas as runs.** O ruído entre runs é grande no nível de família: com configuração idêntica, `half_harmonic` saiu 0,991 numa run e 0,918 na outra. Diferenças de família menores que cerca de 0,07, ou globais de cerca de 1 ponto percentual em DEVELOPMENT, não devem ser interpretadas isoladamente.
- **Protocolos diferentes.** Runs com Kratzer no treino e sem ele avaliam populações ID diferentes (135 contra 130 Hamiltonianos no DEV). Na tabela, a coluna "OOD não visto" inclui Kratzer nas runs sem Kratzer.
- **A janela regenera os dados.** New-Schrodinger, v6 com janela e v8 têm domínios, malhas e rótulos diferentes. Mesmo com o mesmo número de Hamiltonianos, essas comparações não são pareadas.
- **Código do FULL New-Schrodinger.** Foi reconstruído a partir da pasta atual, e não do código exato que gerou a run: `s00`–`s02` foram editados em 09/10.
- **Métricas.** Todas vêm de `per_state_output_metrics.csv` (`category == "ID_test"`). As trajetórias vêm de `training_history.csv`.
- **Nenhuma run pode ser aceita cientificamente.** O critério `matched_direct_and_gnn_mlp_baselines` está fixo em `False` em `s07_reports.py`.
- **A perda de reflexão ficou desativada** (exatamente 0) em todas as runs com microbatch 2, por uma limitação em `collate_training_records`.

## 8. Experimentos que confirmariam as causas

1. ~~Acompanhar a V12 FULL~~ — feito: sem queda no `E_subspace_6`; hipótese do preset refutada (seção 4.3). O próximo passo é medir a **variância entre runs**: repetir a STANDARD V12 com seeds 17 e 29 e, se possível, uma retomada forçada no meio do `D_jepa_latent`, para ver se a retomada sozinha reproduz a queda.
2. **Ablação do preset** (prioridade baixa depois da V12 FULL): só `model_nodes` (257 → 513), só `hidden` (128 → 160), e dilatações escaladas com o número de nós. A V12 FULL mostrou que o preset FULL, por si só, não causa a queda.
3. **Ablação v6 × v8 em STANDARD com o mesmo dataset:** por exemplo, o código v8 com `primary_output=direct_topology` e a janela da v6. Isso separaria o efeito da janela +60% do efeito das perdas e do seletor na regressão da forma de onda.
4. **Mesmo checkpoint, duas regras de publicação:** sempre o Ritz inicial × `physics_selected`, sem retreinar.
5. **Pelo menos 3 seeds** (7, 17 e 29) para a STANDARD v6 e a STANDARD V12.
