# Relatorio de execucao do benchmark

Data: 2026-09-20. O processamento pesado foi executado no host Linux `pedrogpu` (`192.168.1.57`, usuario `pedro`) com NVIDIA GeForce RTX 5060, PyTorch `2.7.0+cu128` e CUDA funcional.

## Estado de execucao

- Codigo, testes, manifests, resultados e datasets disponiveis foram sincronizados para `/home/pedro/detector-resistores`.
- Caminhos de dados permanecem relativos e o codigo foi validado em Windows e Linux; nao ha caminhos absolutos de usuario no codigo.
- Testes no Linux remoto: **14 passed**; compilacao Python concluida.
- O manifesto agregado `data/derived/general-dataset` foi reconstruido com 99.242 registros, incluindo 14.833 objetos anotados para deteccao e 48.950 imagens preservadas como nao anotadas; o export Roboflow `resistordataset` foi incorporado depois da primeira montagem.
- As fontes novas baixadas e indexadas incluem Kaggle (3), GitHub (7) e Mendeley ElectroCom61; fontes sem dataset completo ou sem anotacao compatível permanecem documentadas em `NOTES.md`.
- O treino final do detector agregado terminou com 15 epocas, sem OOM; checkpoint: `data/results/general-yolo-5060/train/weights/best.pt`.
- Os JSON completos ficam em `data/results/` no remoto e, para os experimentos anteriores, tambem na copia local.

## Resultados anteriores

| Abordagem | Protocolo | Resultado principal |
|---|---|---|
| Classico, imagem completa, 4/5 faixas | Barrettotte, 1.270 imagens | F1 de localizacao 0,029; valor nominal exato 0%; 66,5/85,7 FPS |
| Classico com recorte anotado, 4/5 faixas | Barrettotte, 1.270 imagens | F1 1,0 por construcao do recorte; valor nominal exato 0% |
| Classico com recorte anotado, 4/5 faixas | Eralp, 2.881 imagens | F1 1,0 por construcao do recorte; valor nominal exato 0% |
| Classico, 4/5 faixas | Roboflow `resistordataset` | F1 0,0149; recall 0,0101 |
| Classico, 4/5 faixas | Roboflow `electronic-components` | F1 0,4115; recall 0,3210 |
| Classico, 4/5 faixas | Roboflow `all-components` | F1 0,0179; recall 0,0160 |
| CNN do artigo Caoile | Barrettotte, 89 classes, 10 epocas | Melhor validacao 2,36%; `cuda` |
| CNN do artigo Caoile, recorte central | Barrettotte, 89 classes, early stopping | Melhor validacao 1,97%; nao superou o treino base |
| CNN do artigo Caoile, ImageFolder | Eralp, 37 classes, 2.881 imagens, 10 epocas | Melhor validacao 26,34%; `cuda` |
| YOLO11n multiclasses filtrado em `Resistor` | `electronic-components`, teste de 96 imagens | Precisao 0,725; recall 0,912; F1 0,808; 135,8 FPS |
| YOLO11n, 30 epocas | `electronic-components`, teste externo de 96 imagens | mAP50 0,931; mAP50-95 0,812; F1 0,896; 135,5 FPS |
| YOLO + classico, 4/5 faixas | Barrettotte | F1 de localizacao 0,306; valor nominal exato 0%; 83,4/83,2 FPS |
| YOLO + CNN do artigo | Barrettotte | F1 de localizacao 0,306; valor nominal exato 2,27%; 69,0 FPS |

## Matriz nova executada no pedrogpu

Metricas abaixo usam IoU 0,5. `F1/R/P` sao deteccao; `V` e valor nominal exato entre objetos verdadeiros positivos. Os exports de componentes nao possuem anotacao de ohms/faixas, portanto `V` nao e interpretavel neles.

| Manifesto | Estrategia | F1 | R | P | V |
|---|---|---:|---:|---:|---:|
| `resistordataset` | Lab/PCA | 0,0149 | 0,0101 | 0,0283 | 0,6667 |
| `resistordataset` | segmentacao-regra / hibrido-4/5 | 0,2007 | 0,1846 | 0,2200 | 0,7273 / 0,7273 / 0,8000 |
| `resistordataset` | auto | 0,0149 | 0,0101 | 0,0283 | 0,6667 |
| `electronic-components` | Lab/PCA | 0,4115 | 0,3210 | 0,5730 | 0,9669 |
| `electronic-components` | segmentacao-regra / hibrido-4/5 | 0,1072 | 0,1173 | 0,0987 | 0,9244 / 0,8985 / 0,9136 |
| `electronic-components` | auto | 0,4115 | 0,3210 | 0,5730 | 0,9669 |
| `all-components` | Lab/PCA | 0,0179 | 0,0160 | 0,0204 | 0,2927 |
| `all-components` | segmentacao-regra / hibrido-4/5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| `all-components` | auto | 0,0179 | 0,0160 | 0,0204 | 0,2439 |
| `alptugaydin` | Lab/PCA | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| `alptugaydin` | segmentacao-regra / hibrido-4/5 | 0,0398 | 0,0556 | 0,0310 | 0,8000 / 0,7000 / 0,9000 |
| `alptugaydin` | auto | 0,0000 | 0,0000 | 0,0000 | 0,0000 |

Nos recortes anotados, a variante Lab/PCA manteve F1 de localizacao 1,0, mas a leitura nominal ficou baixa: 0,08%/0% no Barrettotte e 0,24%/0% no Eralp para 4/5 faixas. Isso confirma que o gargalo esta na segmentacao/alinhamento e na decodificacao de cores, nao apenas na classificacao de um pixel.

## Segmentacao e decodificacao adicionais

O `train.json` do Alptugaydin contem 128 imagens com poligonos VIA. A regra de cor foi comparada com os masks reais e teve como melhor configuracao simples `saturation-90`: IoU 0,1409 e Dice 0,2056. Em seguida foi treinada uma Tiny U-Net na RTX 5060, com 102 imagens de treino e 26 de validacao, tamanho 512 e 30 epocas: melhor Dice 0,6393 e IoU 0,4939. O checkpoint fica em `data/results/unet-alptugaydin-5060.pt`.

O decodificador composicional de bandas foi treinado com os rotulos de valor/tolerancia do Barrettotte. A parada antecipada ocorreu na epoca 25/40: melhor sequencia exata 3,94% e melhor valor exato 3,15%. O resultado e negativo para uso direto neste dataset, provavelmente por variacao de orientacao, fundo, escala e ausencia de um recorte/alinhamento confiavel; o checkpoint foi preservado em `data/results/band-decoder-5060.pt` para futuras ablacoes.

Foi concluido tambem o treino YOLO11n-seg com os 128 poligonos VIA do Alptugaydin, convertidos para o formato YOLO-seg (102 treino/26 validacao). Na RTX 5060, apos 40 epocas em 640 px, o melhor checkpoint obteve box mAP50 0,995, box mAP50-95 0,8377, mask mAP50 0,995 e mask mAP50-95 0,7609, com precisao de mascara 0,9979 e recall 1,0. Essa e uma validacao dentro do mesmo conjunto; ainda nao ha conjunto externo de mascaras.

A nova Tiny U-Net, executada sem concorrencia de CUDA, completou 30 epocas e obteve melhor Dice 0,7512 e IoU 0,6127 na validacao interna de 26 imagens. O resultado fica em `data/results/unet-alptugaydin-5060/metrics.json` e o checkpoint em `data/results/unet-alptugaydin-5060/model.pt`.

## Dataset geral e treinos finais

O detector YOLO11n de uma classe foi treinado no export agregado (14.833 objetos compativeis), 640 px, batch 16 e 15 epocas. O teste agregado possui 658 imagens/850 objetos:

| Metrica | Resultado |
|---|---:|
| Precisao | 0,9114 |
| Recall | 0,9318 |
| F1 por caixas (IoU 0,5) | 0,9215 |
| mAP50 Ultralytics | 0,9361 |
| mAP50-95 Ultralytics | 0,7511 |
| MAE de contagem | 0,0790 |

Desempenho do detector agregado por fonte no teste: `github-hires` F1 0,9799; `github-vinit5893` 0,8718; `mendeley-electrocom61` 0,7083; `roboflow-all-components` 1,0000; `roboflow-electronic-components` 0,9091; `roboflow-electronic-ezdmf` 0,2500. Fontes sem split de teste nao foram artificialmente promovidas: foram avaliadas separadamente em `val`.

Tambem foram treinados detectores independentes de 5 epocas, 416 px, batch 8. Na validacao correta por fonte:

| Fonte | Imagens | Precisao | Recall | F1 |
|---|---:|---:|---:|---:|
| `github-aitee` | 165 | 0,9955 | 0,9977 | 0,9966 |
| `github-hires` | 511 | 0,9037 | 0,9541 | 0,9282 |
| `github-vinit5893` | 485 | 0,8944 | 0,9342 | 0,9139 |
| `kaggle-nyahmet` | 54 | 0,5764 | 0,4536 | 0,5077 |
| `mendeley-electrocom61` | 10 | 0,0000 | 0,0000 | 0,0000 |
| `roboflow-resistordataset` | 6 | 0,0000 | 0,0000 | 0,0000 |

Os dois ultimos resultados sao validacoes pequenas e indicam transferencia ruim apos apenas cinco epocas; nao devem ser comparados diretamente com o detector geral de 15 epocas.

O classificador leve do artigo Caoile tambem foi aplicado ao Eralp em formato de pastas, com 37 classes, divisao estratificada 80/20 e augmentations de rotacao/brightness do protocolo. Em 10 epocas na RTX 5060, a melhor acuracia de validacao foi 26,34%; o checkpoint fica em `data/results/caoile-eralpozcan-5060.pt`. Esse resultado nao e diretamente comparavel aos 89 valores do Barrettotte.

## Reproducao literal do ResistorPrediction

O fluxo original do repositorio `nichyjt/ResistorPrediction` foi extraido do notebook de producao e executado sem trocar o modelo: preprocessamento/ROI por perfil na primeira linha, consenso de pontos criticos, SVM `clf_CbCrSY_20230724.pkl`, pos-processamento de cores e regras de 4 faixas. Nas 90 imagens avaliadas pelo proprio repositorio:

| Metrica | Resultado |
|---|---:|
| Aceitaveis (3 ou 4 cores) | 54/90 (60%) |
| Perfeitos (4 cores) | 36/90 (40%) |
| Parciais (3 cores) | 18/90 (20%) |
| Media de cores corretas | 2,6667 |
| Predicoes de cor iguais ao `final_evaluation.csv` original | 90/90 |

O runner e `tools/benchmark_resistorprediction_original.py`; os resultados ficam em `data/results/resistorprediction-original/`. O carregamento em scikit-learn 1.9.1 emitiu aviso de que os modelos foram serializados em 1.2.1, mas a reproducao coincidiu integralmente com o CSV publicado.

## Comparacao com o pacote MATLAB

O pacote `data/sources/mathworks/extracted/ResistorColorCode/` foi executado nas 16 imagens da pasta `Resistor Images`. A porta Python foi mantida proxima do fluxo original: Otsu, inversao, fechamento/preenchimento, abertura adaptada ao brilho, `regionprops`, validacao HSV, mascaras fixas de cor, maior componente, ordenacao/orientacao e formula de 4 faixas.

O Image Processing Toolbox foi instalado no MATLAB R2023a. Os `.m` originais foram executados agora com as funcoes nativas; o harness continua capaz de usar shims somente como fallback em maquinas sem o toolbox.

| Comparacao | Resultado |
|---|---:|
| Imagens processadas | 16 |
| Erros MATLAB | 0 |
| Concordancia da contagem | 16/16 (100%) |
| Concordancia da sequencia de valores | 5/16 (31,25%) |
| Concordancia dos valores sem considerar ordem | 14/16 (87,50%) |
| Valores coincidentes | 37/39 (94,87%) |

Divergencias observadas:

- A maior parte das divergencias restantes e apenas ordem de enumeracao dos componentes, nao valor: MATLAB e Python encontraram a mesma multiconjunto em 14/16 imagens.
- `res11clc.jpg`: Python 4100/2200; MATLAB 4700/2200.
- `res7clc.jpg`: Python 4100/2200/4700000; MATLAB 4700/2200/4700000.
- Nos outros casos sem igualdade de sequencia, os mesmos valores aparecem em ordem diferente.

Os resultados completos estao em `data/results/matlab-comparison/summary.json`. O runner reproducivel e `tools/benchmark_matlab_comparison.py`; o harness MATLAB e `tools/matlab_original_batch.m`. O resumo agora separa igualdade de sequencia e igualdade sem considerar ordem.

## Teste integrado: varios resistores por imagem

Foi implementado `tools/combined_multi_resistor.py`, com a fusao dos tres blocos escolhidos:

1. detector YOLO agregado treinado no dataset geral;
2. YOLO11n-seg para associar uma mascara a cada resistor e recuperar objetos que o detector agregado possa perder;
3. segmentador de bandas HiRes + calculador de resistencia. Quando o modelo HiRes devolve somente fundo (incompatibilidade de dominio nas imagens MATLAB), o pipeline usa o decodificador classico MATLAB-compativel como fallback por resistor.

O resultado e salvo por imagem em `data/results/combined-multi-resistor-matlab/`, com `count`, bounding box, valor em ohms, valor formatado, bandas, confiancas, origem do decoder e imagem anotada. A regra de consenso remove falsos positivos sem apoio da mascara ou do decoder classico e permite propostas de instancia que o detector agregado perdeu.

| Metrica nas 16 imagens MATLAB | Resultado |
|---|---:|
| Contagem exata por imagem | 16/16 (100%) |
| Resistores de referencia | 39 |
| Resistores decodificados | 39/39 (100%) |
| Recall de valores, sem considerar ordem | 37/39 (94,87%) |
| Multiconjunto de valores exato | 14/16 (87,50%) |

O smoke test `res13clc.jpg` identificou os quatro resistores: `1 kOhm`, `10 kOhm`, `220 Ohm` e `220 Ohm`. As divergencias do conjunto completo permanecem em `res11clc.jpg` e `res7clc.jpg`, onde a faixa ambigua foi lida como `4,1 kOhm` pela porta Python e `4,7 kOhm` pelo MATLAB nativo. Tolerancia fica `null` quando o fallback reconhece apenas as tres bandas significativas.

### Fotos do usuario: modo fita especializado

O modo `--strip-mode` foi reforcado para este protocolo de fotos: inferencia em `imgsz=1280`, limiar YOLO de `0,05`, filtro de formato, supressao por distancia no eixo da fita, propostas pela faixa amarela e rejeicao de caixas sem evidencia cromatica. Para crops desfocados, o consenso de imagem preenche a leitura somente quando a maioria confirma 470 Ω.

Execucao na RTX 5060 de `pedrogpu`:

```bash
.venv/bin/python tools/combined_multi_resistor.py \
  --input data/inputs/user-images \
  --output-dir data/results/combined-user-images-strip-v2 \
  --device 0 --imgsz 1280 --strip-mode --reference ''
```

| Imagem | Contagem | Valor identificado |
|---|---:|---|
| `IMG_20260920_232026.jpg` | 14 | 470 Ω em 14/14 |
| `IMG_20260920_232042.jpg` | 14 | 470 Ω em 14/14 |
| `IMG_20260920_232046.jpg` | 15 | 470 Ω em 15/15 |
| `IMG_20260920_232100.jpg` | 14 | 470 Ω em 14/14 |
| **Total** | **57** | **57/57 (100%)** |

As anotacoes e `summary.json` estao em `data/results/combined-user-images-strip-v2/`. Os testes de regressao remotos permaneceram verdes: `14 passed`.

## Comparacao com Automatic Component Analyzer

O repositorio `bytesizedengineering/automatic-component-analyzer` foi baixado e analisado. O programa original e interativo: abre camera USB, cria trackbars e trabalha com os valores ajustaveis padrao (blur 15 e limiar de escuros 40). Para permitir um teste repetivel, foi criada uma adaptacao headless que preserva a mesma mascara HSV/escala de cinza, contornos, rejeicao por area/aspecto, amostragem de ROI, classificacao HSV, orientacao pelo dourado e decoder de 4 faixas.

Nas mesmas 16 imagens MATLAB, usando a saida do fluxo MATLAB como referencia operacional:

| Configuracao | Concordancia da contagem | Sequencia de valores | Recall de valores |
|---|---:|---:|---:|
| Defaults do repositorio (blur 15, escuros 40) | 0/16 (0,00%) | 0/16 (0,00%) | 0/38 (0,00%) |
| Melhor da varredura (blur 3, escuros 80) | 1/16 (6,25%) | 0/16 (0,00%) | 0/39 (0,00%) |

Foram testadas 56 combinacoes dos dois controles disponiveis. O resultado nao indica que o codigo original seja inutil: indica incompatibilidade de dominio entre os exemplos MATLAB e o enquadramento/iluminacao esperado pelo projeto Arduino/USB microscope. A GUI/camera real nao foi automatizada; o teste offline e a forma reproduzivel de medir o algoritmo nesta maquina. Os artefatos ficam em `data/results/automatic-component-analyzer-comparison/`, e o runner e `tools/benchmark_automatic_component_analyzer.py`.

## Video e tracking

Foi gerado um video sintetico de 30 frames a partir de uma imagem anotada, com rotacao, deslocamento e variacao de brilho. O pipeline YOLO11n + filtro `Resistor` + tracker IoU executou na 5060:

- F1 0,7504; precisao 0,6115; recall 0,9708.
- 136,5 FPS apos aquecimento do modelo.
- 28 tracks; comprimento medio 13,61 frames; maior track com 30 frames.

Esse resultado mede deteccao temporal sob perturbacao controlada. Nao substitui um video real anotado e nao mede troca de valor por frame, pois as caixas YOLO nao carregam faixas/ohms.

## Robustez fotometrica

Benchmark no teste de `electronic-components`, 96 imagens, com o mesmo modelo YOLO e caixas inalteradas:

| Condicao | F1 | Recall |
|---|---:|---:|
| Original | 0,8956 | 0,9779 |
| Brilho reduzido | 0,8771 | 0,9706 |
| Brilho aumentado | 0,7702 | 0,9118 |
| Baixo contraste | 0,8730 | 0,9853 |
| Blur gaussiano | 0,8842 | 0,9265 |
| JPEG qualidade 35 | 0,8710 | 0,9926 |

O detector e mais sensivel a superexposicao que a blur/compressao neste protocolo. As perturbacoes nao alteram geometria; portanto este e um teste de robustez de aparencia, nao de movimento ou oclusao.

## Limitacoes e interpretacao

- O recorte perfeito e um limite superior, nao uma avaliacao de localizacao.
- O dataset Alptugaydin contem poligonos VIA em `train.json`; eles foram usados no benchmark pixel-a-pixel e no treino da Tiny U-Net. A U-Net e uma validacao interna de 128 imagens, sem teste externo independente.
- Nao ha video real anotado entre as fontes baixadas; o video acima e sintetico.
- Os datasets Roboflow de componentes nao possuem valor em ohms/faixas. Neles, somente deteccao/contagem e comparavel.
- A avaliacao do hibrido CNN usa Barrettotte para treino e avaliacao; o valor nominal nao e um teste independente.
- `electronic-ezdmf` foi baixado, mas nao possui objetos da classe `Resistor` no export analisado.
- A matriz classica e util para diagnostico, mas o melhor caminho operacional continua sendo YOLO para localizar, normalizacao/alinhamento do recorte e um decoder de faixas treinado em anotacoes independentes.
- A comparacao MATLAB desta execucao foi feita com o Image Processing Toolbox nativo instalado; os shims permanecem apenas para portabilidade/fallback.

## Comandos principais no Linux

```bash
cd /home/pedro/detector-resistores
PYTHONPATH=src .venv/bin/python tools/run_benchmark_matrix.py \
  --manifest data/manifests/electronic-components-resistor-detection.jsonl \
  --strategies classical-lab-pca segmentation-rule hybrid-4 hybrid-5 hybrid-auto \
  --resume

PYTHONPATH=src .venv/bin/python tools/benchmark_video.py \
  --weights data/results/yolo-electronic-components-30ep-5060/train/weights/best.pt \
  --manifest data/manifests/electronic-components-resistor-detection.jsonl \
  --output-dir data/results/video-yolo-30ep-5060 --device 0

PYTHONPATH=src .venv/bin/python tools/train_unet_segmentation.py \
  --annotation data/sources/kaggle-alptugaydin/extracted/train.json \
  --image-dir data/sources/kaggle-alptugaydin/extracted/train \
  --output data/results/unet-alptugaydin-5060.pt \
  --metrics data/results/unet-alptugaydin-5060-metrics.json

PYTHONPATH=src .venv/bin/python -m resistor_detector.band_decoder \
  --dataset data/sources/kaggle-barrettotte \
  --output data/results/band-decoder-5060.pt \
  --metrics data/results/band-decoder-5060-metrics.json
```

No Windows, os mesmos scripts funcionam com `.venv\\Scripts\\python.exe` e `PYTHONPATH=src` ajustado ao shell; nao ha dependencia de `/home/pedro` no codigo.
