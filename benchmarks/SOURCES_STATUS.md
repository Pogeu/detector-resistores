# Status dos materiais baixados

Data da coleta: 2026-09-19.

## Disponíveis localmente

| Fonte | Status | Conteúdo local |
| --- | --- | --- |
| Zenodo 20699559 | Artigo, sem dataset anexado | O PDF já existente foi copiado para `data/sources/zenodo/Caoile-paper.pdf`. A API do registro lista somente esse PDF. O artigo descreve 1.270 imagens de resistores de 4/5 bandas e 89 valores; isso coincide fortemente com `barrettotte/resistors` (1.270 imagens e 89 valores distintos), mas o artigo não informa URL, DOI ou repositório do dataset, então a identidade não foi confirmada documentalmente. |
| Kaggle Eralp Ozcan | Completo | `data/sources/kaggle-eralpozcan/resistor-dataset.zip` validado e extraído. 2.881 imagens JPG. |
| Kaggle Alptugaydin | Completo | ZIP validado e extraído. 105 imagens JPG e `train.json`; a anotação contém 128 imagens. |
| Kaggle Barrettotte | Completo | ZIP validado e extraído. 1.270 imagens JPG, com `4-band.json` e `5-band.json`. |
| MATLAB File Exchange | Completo | `data/sources/mathworks/resistor-color-code-detector.zip` validado e extraído. 41 arquivos. |
| Automatic Component Analyzer | Completo | Clone Git raso em `data/sources/github-automatic-component-analyzer/`. |
| ResistorPrediction | Completo | Clone Git raso em `data/sources/github-resistorprediction/`, incluindo imagens, rótulos e notebooks/scripts. |
| DirençNet / ResistorNet | Referência local | A página Scribd foi salva em `data/sources/scribd-direncnet/page.html`. O documento completo não foi baixado porque a plataforma restringe o conteúdo; não foi feito bypass. |
| Roboflow Harish — `resistordataset` | Completo | `data/sources/roboflow-resistordataset/dataset-yolov5pytorch.zip`, validado e extraído. 135 imagens na versão exportada. |
| Roboflow ABC — `electronic-ezdmf` | Completo | `data/sources/roboflow-electronic-ezdmf/dataset-yolov5pytorch.zip`, validado e extraído. 239 imagens na versão exportada. |
| Roboflow Samu — `electronic-components-d6uul` | Completo | `data/sources/roboflow-electronic-components/dataset-yolov5pytorch.zip`, validado e extraído. 9.850 imagens na versão exportada. |
| Roboflow NED — `all-components` | Completo | `data/sources/roboflow-all-components/dataset-yolov5pytorch.zip`, validado e extraído. 35.949 itens no pacote exportado. |

## Não disponíveis sem credencial ou exportação autorizada

| Fonte | Resultado |
| --- | --- |
| Kaggle `skender85/resistor-evaluationv5` | A página HTML foi salva em `data/sources/kaggle-skender85/notebook-page.html`. Os endpoints públicos testados para baixar o notebook retornaram 404. O código exige a API/CLI do Kaggle e credencial válida. |

As cópias Kaggle foram testadas com `python -m zipfile -t` antes da extração.
Os ZIPs originais foram preservados; as pastas `extracted/` são cópias de trabalho.
