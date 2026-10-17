# Inventario das fontes e abordagens

Data da analise: 2026-09-20.

Este documento separa as fontes listadas em `NOTES.md` das abordagens de deteccao, segmentacao e leitura de valor que elas representam. Tambem registra o que foi baixado, o que foi reproduzido no projeto e o que ainda nao foi executado literalmente.

## Resumo executivo

As familias de abordagem encontradas foram:

1. **Processamento classico de imagem**: segmentacao por cor, limiarizacao HSV/Lab, contornos, ROI, amostragem de faixas e decodificacao IEC.
2. **Classificacao de imagem com CNN**: uma classe por valor nominal do resistor.
3. **Classificacao de cores com ML classico**: extracao de caracteristicas de cor e SVM/classificador, seguida de regras para montar o valor.
4. **Deteccao de objetos**: YOLO para localizar resistores em cenas com um ou varios componentes.
5. **Segmentacao semantica/por instancia**: mascaras de resistor para recorte mais preciso antes da leitura.
6. **Hibridos**: detector para localizar, recorte/alinhamento para normalizar e decoder classico ou CNN para ler as faixas.
7. **Video**: deteccao por quadro e associacao temporal por IoU/tracking.

As abordagens principais foram implementadas e comparadas. Ainda nao foi feita a reproducao literal de cada notebook ou GUI das fontes externas.

## Fontes individuais

### 1. Zenodo 20699559 e PDF de Caoile

Fonte: <https://zenodo.org/records/20699559>

Arquivo local: `references/papers/JETSUR_01_01_009_(C.K.L. Caoile).pdf` e `data/sources/zenodo/Caoile-paper.pdf`.

**O que a fonte propoe**

- CNN leve customizada para classificar o valor de resistores de 4 e 5 faixas.
- Imagens redimensionadas para 64 x 64, normalizadas e aumentadas.
- Classes organizadas por valor nominal.
- Interface de webcam com uma ROI guiada para ajudar o usuario a posicionar o resistor.
- O artigo propoe YOLO-CNN como extensao futura: YOLO localiza e uma CNN classifica as faixas.

**Dados**

- O registro Zenodo acessivel contem o artigo, mas nao apresenta um arquivo de dataset separado.
- O PDF e internamente inconsistente: o resumo/tabela falam em aproximadamente 1.200 imagens, enquanto a secao de resultados informa 1.270 imagens e 89 classes. O Barrettotte possui exatamente 1.270 imagens e 89 valores, por isso foi adotado como o substituto operacional mais provavel, mas a identidade entre os dois conjuntos nao foi comprovada documentalmente.

**No projeto**

- Reproduzida a ideia da CNN leve com o Barrettotte: `src/resistor_detector/caoile_cnn.py`.
- Treino base e treino com recorte central executados na RTX 5060.
- A mesma CNN foi treinada no Eralp em ImageFolder: 37 classes, 2.881 imagens, 80/20 estratificado, 10 epocas e melhor validacao 26,34%; checkpoint em `data/results/caoile-eralpozcan-5060.pt`.
- A variante de 50 epocas teve melhor validacao de aproximadamente 1,97%; o treino base chegou a aproximadamente 2,36% de erro de validacao.
- A localizacao YOLO-CNN foi implementada como benchmark hibrido, mas a leitura de valor ainda depende da qualidade do recorte.

**Nao feito literalmente**

- Recriar a webcam/ROI grafica do artigo.
- Treinar com o dataset original do artigo, pois ele nao foi anexado ao registro.

### 2. Kaggle Eralp Ozcan - Resistor Dataset

Fonte: <https://www.kaggle.com/datasets/eralpozcan/resistor-dataset>

**Tipo**: imagens rotuladas de resistores para classificacao/leitura de valor.

**Download**: completo em `data/sources/kaggle-eralpozcan/`; 2.881 imagens JPG validadas e extraidas. Sincronizado para `pedrogpu`.

**Abordagens aplicaveis**

- Classificacao por valor nominal.
- Recorte de ROI seguido de leitura de cores.
- Treino de CNN ou decoder de faixas.

**No projeto**

- Usado no benchmark classico com recorte anotado e no diagnostico de Lab/PCA.
- O recorte perfeito foi avaliado como limite superior de segmentacao.

**Nao feito**

- Nao foi feita uma avaliacao independente fora do conjunto Eralp; o treino especifico agora foi executado, mas continua sendo validacao interna.
- Nao ha anotacao de caixas/mascaras para treinar detector de objetos com ele.

### 3. Kaggle Alptugaydin - Resistor Images Dataset for Segmentation

Fonte: <https://www.kaggle.com/datasets/alptugaydin/resistor-images-dataset-for-segmentation>

**Tipo**: imagens de resistor com anotacoes VIA, incluindo poligonos em `train.json`.

**Download**: completo em `data/sources/kaggle-alptugaydin/`; ZIP validado, imagens extraidas e `train.json` presente. Sincronizado para `pedrogpu`.

**Abordagens aplicaveis**

- Segmentacao por regra de cor.
- Segmentacao semantica com U-Net.
- Recorte do resistor antes da leitura de faixas.

**No projeto**

- Feito benchmark pixel-a-pixel com varredura de limiares HSV/saturacao.
- Melhor regra simples: IoU 0,1409 e Dice 0,2056.
- Tiny U-Net treinada com 102 imagens de treino e 26 de validacao.
- Melhor U-Net: Dice 0,6393 e IoU 0,4939.
- Script: `tools/benchmark_segmentation_masks.py`.
- Treino: `tools/train_unet_segmentation.py`.

**Nao feito**

- Avaliacao em teste externo independente, pois o material disponivel foi tratado como um conjunto anotado de 128 imagens.

**Treino adicional executado**

- Os poligonos VIA foram convertidos por `tools/prepare_yolo_segmentation.py` para YOLO-seg.
- YOLO11n-seg treinado por 40 epocas na RTX 5060: mask mAP50 0,995 e mask mAP50-95 0,7609; box mAP50 0,995.
- O resultado e validação interna 102/26, nao evidencia generalizacao externa.

### 4. Kaggle Barrettotte - Resistors

Fonte: <https://www.kaggle.com/datasets/barrettotte/resistors>

**Tipo**: imagens rotuladas por pastas de resistores de 4 e 5 faixas, com arquivos JSON de anotacao.

**Download**: completo em `data/sources/kaggle-barrettotte/`; 1.270 imagens JPG, `4-band.json` e `5-band.json`. Sincronizado para `pedrogpu`.

**Abordagens aplicaveis**

- CNN de classificacao de valor.
- Decoder composicional de cores.
- Leitura classica por amostragem de faixas.
- Benchmark de recorte/alinhamento.

**No projeto**

- Principal conjunto para comparar classico, Lab/PCA, recorte, CNN e hibridos.
- Usado no treino da CNN de Caoile.
- Usado no treino experimental do decoder composicional de bandas.
- Usado para gerar o limite superior com recorte anotado.

**Limitacao**

- O desempenho dentro do mesmo dominio nao representa generalizacao independente; o conjunto tem fundos, orientacoes e composicoes muito especificas.

### 5. Kaggle `skender85/resistor-evaluationv5`

Fonte: <https://www.kaggle.com/code/skender85/resistor-evaluationv5/notebook>

**Tipo**: notebook de avaliacao, nao um dataset independente claramente exposto pelo link.

**Download/status**

- A pagina HTML foi salva em `data/sources/kaggle-skender85/notebook-page.html`.
- Tentativas de obter o notebook por endpoints publicos retornaram 404.
- A recuperacao por API/CLI exige credencial Kaggle valida.

**Nao feito**

- O notebook original nao foi executado.
- Nenhum dataset adicional foi obtido a partir dele.

**O que o usuario precisa fornecer se quiser a reproducao literal**

- Export/download autorizado do notebook `.ipynb` ou credencial Kaggle configurada.

### 6. MATLAB File Exchange - Resistor Color Code/Value Detector

Fonte: <https://www.mathworks.com/matlabcentral/fileexchange/73099-resistor-color-code-value-detector-from-image>

**O que a fonte propoe**

- Detectar resistores em uma imagem por segmentacao.
- Separar as faixas usando mascaras de cor.
- Classificar as cores e decodificar um resistor de 4 faixas.
- Retornar valor em ohms, mas nao a tolerancia.
- Interface grafica `MainGUI.m` com imagens de exemplo.

**Download**: ZIP de 11,4 MB completo em `data/sources/mathworks/resistor-color-code-detector.zip`, validado e extraido.

**No projeto**

- A familia de algoritmo foi reimplementada em Python/OpenCV: candidatos por mascara/contorno, amostragem de cores e decoder IEC.
- Foram adicionadas variantes HSV, Lab/PCA, segmentacao por regra e hibridos.
- Foi criada uma porta de compatibilidade com os mesmos limiares e ordem do pacote MATLAB. Com o Image Processing Toolbox nativo instalado, a contagem coincidiu em 16/16; os valores coincidiram sem considerar ordem em 14/16 e 37/39 valores.

**Nao feito literalmente**

- Execucao do `MainGUI.m` dentro do MATLAB.
- Comparacao byte a byte contra as mesmas imagens de exemplo do pacote.

O Image Processing Toolbox nativo foi usado nas 16 imagens; os shims permanecem como fallback para Windows/Linux sem o produto instalado.

### 7. GitHub - Automatic Component Analyzer

Fonte: <https://github.com/bytesizedengineering/automatic-component-analyzer>

**O que a fonte propoe**

- Captura ao vivo de camera/USB microscope.
- Ajuste manual de limiares HSV por trackbars.
- Mascara para cores saturadas, branco/prata e cores escuras.
- Contornos para encontrar ate quatro faixas.
- Ordenacao horizontal e tentativa de inverter a leitura quando dourado aparece na extremidade errada.
- Decoder de 4 faixas que tambem retorna tolerancia.
- Visualizacao ao vivo com caixas e valor.

**Download**: clone completo em `data/sources/github-automatic-component-analyzer/`.

**No projeto**

- A estrategia esta documentada e parcialmente coberta pelas implementacoes classicas de HSV, contornos, ordenacao e decoder.
- O projeto atual generaliza o decoder para 4/5 faixas e adiciona Lab/PCA, rejeicao e benchmark.

**Teste especifico**

- Foi criado `tools/benchmark_automatic_component_analyzer.py` e o adaptador headless `src/resistor_detector/automatic_component_analyzer.py`.
- Nas 16 imagens do pacote MATLAB, os defaults do repositorio deram 0/16 em contagem, 0/16 em sequencia e 0/39 valores coincidentes com a referencia MATLAB nativa.
- Uma varredura de 56 combinacoes dos dois controles do projeto encontrou blur 3/limiar 80 como melhor: 1/16 contagens, 0/16 sequencias e 0/39 valores.
- Isso mede transferencia para o dominio das imagens MATLAB; nao substitui uma avaliacao com camera USB e iluminacao esperadas pelo repositorio.

**Nao feito literalmente**

- Executar o `software/main.py` com uma camera real e os trackbars.
- Reproduzir a interface grafica Arduino Uno Q/USB microscope.

### 8. Roboflow Harish - `resistordataset`

Fonte: <https://universe.roboflow.com/harish-madhavan/resistordataset>

**Tipo**: deteccao de objetos com uma classe `Resistor`.

**Download**: export YOLOv5 completo em `data/sources/roboflow-resistordataset/`; 135 imagens no export usado. Sincronizado para `pedrogpu`.

**No projeto**

- Usado na matriz de benchmark de deteccao/classico.
- A classe e adequada para treinar YOLO, mas nao carrega valor em ohms nem cores de faixa.

**Nao feito**

- Nao foi escolhido como treino YOLO principal; o treino longo foi feito no `electronic-components`, maior e mais variado.

### 9. Roboflow ABC - `electronic-ezdmf`

Fonte: <https://universe.roboflow.com/abc-qqksz/electronic-ezdmf>

**Tipo**: deteccao de componentes eletronicos; o export possui 239 imagens e classes de componentes.

**Download**: completo em `data/sources/roboflow-electronic-ezdmf/`.

**No projeto**

- Manifesto preparado e export analisado.
- O material nao apresentou objetos `Resistor` suficientes/uteis no manifesto usado; por isso nao foi usado para treinar o detector final.

### 10. Roboflow Samu - `electronic-components-d6uul`

Fonte: <https://universe.roboflow.com/samu-drioq/electronic-components-d6uul>

**Tipo**: deteccao de componentes, com classe `Resistor` no export utilizado.

**Download**: export YOLO completo em `data/sources/roboflow-electronic-components/`; 9.850 imagens no pacote e 2.794 registros de resistor no manifesto de trabalho. Sincronizado para `pedrogpu`.

**No projeto**

- Foi o conjunto principal para YOLO11n.
- Treino de 5 epocas e treino final de 30 epocas na RTX 5060.
- Melhor modelo final: `data/results/yolo-electronic-components-30ep-5060/train/weights/best.pt`.
- Avaliacao externa: mAP50 0,9306, mAP50-95 0,8120, F1 0,8956 e 135,5 FPS.

### 11. Scribd - DirençNet/ResistorNet

Fonte: <https://pt.scribd.com/document/381301116/DirencNet-ResistorNet>

**O que a fonte descreve**

- Dataset proprio com 37 classes e 3.025 imagens finais.
- Inception-ResNet-v2 com Keras/TensorFlow.
- Data augmentation com rotacoes de 15 graus.
- Divisao informada de 70% treino, 20% teste e 10% validacao.
- Resultado publicado de aproximadamente 82% de acuracia.

**Download/status**

- A pagina HTML/previsualizacao foi salva em `data/sources/scribd-direncnet/page.html`.
- O documento completo e o dataset nao foram baixados porque o Scribd restringe o conteudo.

**Nao feito**

- Reproducao do Inception-ResNet-v2 com os dados originais.
- Comparacao direta com os 82% publicados.

**O usuario precisa fornecer**

- PDF completo autorizado e, se possivel, o dataset DirençNet original.

### 12. Roboflow NED - `all-components`

Fonte: <https://universe.roboflow.com/ned-university-of-engineering-and-technology-5f8dq/all-components>

**Tipo**: deteccao de varios componentes eletronicos.

**Download**: export YOLO completo em `data/sources/roboflow-all-components/`; 35.949 itens no pacote exportado. Sincronizado para `pedrogpu`.

**No projeto**

- Incluido na matriz de deteccao e segmentacao fraca.
- Serve para contar/localizar componentes, mas nao possui rotulos de ohms ou sequencia de cores.

**Nao feito**

- Nao foi usado no treino YOLO final porque o objetivo operacional e detectar especificamente resistores.

### 13. GitHub - ResistorPrediction

Fonte: <https://github.com/nichyjt/ResistorPrediction>

**O que a fonte propoe**

- Pipeline OpenCV + machine learning para resistor de 4 faixas.
- Preprocessamento e deteccao de ROI.
- Extracao de caracteristicas de cor.
- Classificador de cores, incluindo experimentos com SVM e espacos CbCr/HY.
- Regras para transformar as quatro cores no valor de resistencia.
- O proprio repositorio reporta 54/90 resultados aceitaveis, 36 perfeitos e 18 parcialmente corretos.

**Download**: clone completo em `data/sources/github-resistorprediction/`, incluindo notebooks, imagens, modelos `.pkl`, rotulos e CSVs de resultados.

**No projeto**

- A ideia de reduzir o problema a classificacao de cores e regras de valor foi analisada.
- O fluxo dos notebooks foi reproduzido em `tools/benchmark_resistorprediction_original.py`, usando o mesmo SVM `.pkl`, features Cb/Cr/S/Y, heuristicas e regras de valor.
- Nas 90 imagens avaliadas pelo repositorio: 54/90 aceitaveis, 36/90 perfeitas e 18/90 parciais; as 90 predicoes de cor coincidiram com `results/final_evaluation.csv`.

**Nao feito literalmente**

- Executar os notebooks em interface Jupyter, com seus graficos interativos.
- Reproduzir a versao exata do ambiente scikit-learn 1.2.1; a execucao atual usa 1.9.1 e confirmou as saidas publicadas.

## Datasets e arquivos baixados

### Disponiveis localmente e no `pedrogpu`

- Kaggle Eralp Ozcan.
- Kaggle Alptugaydin.
- Kaggle Barrettotte.
- Roboflow `resistordataset`.
- Roboflow `electronic-ezdmf`.
- Roboflow `electronic-components-d6uul`.
- Roboflow `all-components`.
- Pacote MATLAB.
- Clone `automatic-component-analyzer`.
- Clone `ResistorPrediction`.
- PDF Caoile.
- Pagina HTML do Scribd.
- Pagina HTML do notebook Kaggle.

Todos os manifests de trabalho foram conferidos no Linux remoto sem caminhos ausentes.

### Nao obtidos completamente

1. **Dataset do Zenodo/Caoile**: nao existe arquivo separado anexado ao registro acessivel; somente o artigo foi encontrado.
2. **Notebook Kaggle `skender85`**: a pagina foi salva, mas o `.ipynb` exige credencial/export autorizado; os endpoints publicos testados retornaram 404.
3. **PDF/dataset DirençNet completo**: Scribd permite apenas a pagina/previsualizacao sem download autorizado.

Se o usuario fornecer esses arquivos, eles podem ser colocados em `data/sources/` e os scripts de preparacao/manifests poderao ser executados sem alterar o pipeline.

## Matriz de execucao das abordagens

| Abordagem | Implementada no projeto | Executada/medida | Observacao |
|---|---:|---:|---|
| Contornos + mascaras de cor + decoder IEC | Sim | Sim | Melhor como baseline controlado; sensivel a iluminacao/alinhamento |
| HSV/Lab/PCA e variantes de segmentacao | Sim | Sim | Benchmark em varios datasets |
| CNN leve do artigo Caoile | Sim | Sim | Baixa generalizacao no Barrettotte avaliado |
| YOLO11n para localizar resistores | Sim | Sim | Melhor resultado de deteccao |
| YOLO + classico/CNN | Sim | Sim | Hibrido funcional, leitura ainda limitada pelo recorte |
| U-Net com mascaras VIA | Sim | Sim | Melhor que regra simples; ainda sem teste externo |
| YOLO11n-seg com poligonos VIA | Sim | Sim | 40 epocas; mask mAP50 0,995; validacao interna |
| Decoder neural composicional de bandas | Sim | Sim | Resultado baixo; mantido como experimento |
| IoU tracking em video | Sim | Sim | Medido em video sintetico |
| MATLAB original com GUI | Parcial | Sim, batch nativo | 16 imagens; toolbox nativo instalado; GUI ainda nao automatizada |
| Automatic Component Analyzer com camera/trackbars | Parcial | Sim, adaptador headless | 16 imagens MATLAB + varredura de 56 configuracoes; GUI/camera real nao executadas |
| ResistorPrediction com notebooks e SVM original | Parcial | Sim, runner headless | 54/90 aceitaveis; 90/90 predicoes iguais ao CSV original |
| DirençNet Inception-ResNet original | Nao | Nao | Dataset/PDF completo indisponivel |
| Notebook Kaggle `skender85` | Nao | Nao | Requer credencial/export |

## Conclusao

O trabalho ja cobre as familias tecnicas necessarias para o detector: deteccao, segmentacao, classificacao, decodificacao e video. O que permanece pendente e a reproducao fiel de materiais externos bloqueados ou de interfaces/notebooks especificos, nao uma nova familia fundamental de deteccao.

Para a versao operacional, a recomendacao continua sendo:

`YOLO -> recorte/alinhamento -> U-Net ou mascara -> leitura classica/hibrida das faixas -> tracking temporal`
