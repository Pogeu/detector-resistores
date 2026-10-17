# Dataset geral agregado

O dataset geral e um indice com preservacao de procedencia. Ele nao mistura silenciosamente tarefas com espacos de rotulos incompatíveis: cada registro informa fonte, tarefa, caminho original e anotacao disponivel.

## Artefatos no `pedrogpu`

- `data/derived/general-dataset/manifest.jsonl`: todos os registros descobertos.
- `data/derived/general-dataset/object-detection-manifest.jsonl`: objetos compatíveis com deteccao de resistor.
- `data/derived/general-dataset/value-manifest.jsonl`: exemplos com informacao de valor/faixas quando disponivel.
- `data/derived/general-dataset/yolo/`: export normalizado YOLO de uma classe (`resistor`), com `train`, `val` e `test`.
- `data/derived/general-dataset/summary.json`: contagens e fontes usadas na montagem.

## Contagem atual

| Item | Quantidade |
|---|---:|
| Registros totais | 99.242 |
| Objetos para deteccao | 14.833 |
| Classificacao de componente | 25.464 |
| Classificacao de valor | 120 |
| Deteccao legada | 5.490 |
| Segmentacao com mascara | 128 |
| Recortes com valor | 4.257 |
| Imagens sem anotacao compatível | 48.950 |

Fontes com objetos de resistor no export final: `github-hires` (2.844), `github-vinit5893` (5.174), `roboflow-all-components` (2.570), `roboflow-electronic-components` (2.794), `github-aitee` (823), `kaggle-nyahmet` (256), `roboflow-electronic-ezdmf` (54), `mendeley-electrocom61` (54), `roboflow-resistordataset` (132) e o subconjunto `extracted` (132).

## Regra de uso

O treino YOLO geral usa somente `object-detection-manifest.jsonl` convertido para uma classe. Imagens de classificação, recortes de valor e imagens sem anotacao continuam no manifesto para rastreabilidade e para os experimentos especificos; elas nao recebem uma caixa inventada. O leitor de ohms continua sendo avaliado separadamente, porque a maioria das fontes de deteccao nao informa faixas coloridas nem valor nominal.

## Execucao

O treino oficial do detector agregado foi concluido na RTX 5060 de `pedrogpu` em 640 px, 15 epocas e batch 16. No teste agregado, obteve precisao 0,9114, recall 0,9318, F1 0,9215, mAP50 0,9361 e mAP50-95 0,7511. A fila tambem concluiu a Tiny U-Net de mascaras (Dice 0,7512; IoU 0,6127) e seis modelos YOLO de 5 epocas por fonte. Resultados e checkpoints permanecem em `data/results/` no host remoto.
