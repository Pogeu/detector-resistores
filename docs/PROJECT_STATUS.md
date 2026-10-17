# Documentação completa do projeto

Projeto de Processamento de Sinais 2 para contagem e identificação automática do valor de resistores axiais em imagens e vídeos.

**Estado documentado:** 17 de outubro de 2026  
**Autor:** Pedro Nicollas Pereira Azevedo Della Torre Bastos  
**Orientador:** Prof Gabriel Matos Araujo  
**Plataforma de treino:** Linux remoto em pedrogpu, endereço 192.168.1.57, usuário pedro, GPU RTX 5060

## 1. Objetivo

O sistema recebe uma imagem ou um vídeo contendo um ou vários resistores, localiza cada componente, conta quantos existem e informa individualmente:

- caixa ou região do resistor na imagem;
- sequência de faixas de cor quando legível;
- valor nominal em ohms;
- tolerância, quando identificável;
- confiança e, em vídeos, identidade temporal do componente.

O caso de uso principal são fotos de fitas com muitos resistores iguais, como as imagens do usuário em que os componentes deveriam ser identificados como 470 ohms.

## 2. O que foi feito

Até aqui, o projeto passou pelas seguintes etapas:

1. Levantamento de artigos, datasets, códigos e abordagens disponíveis.
2. Download e organização das fontes que puderam ser obtidas legalmente.
3. Implementação de um pacote Python portátil para Windows e Linux.
4. Implementação de decodificação de resistores de 4 e 5 faixas.
5. Implementação de um detector clássico baseado em OpenCV, cor, morfologia e contornos.
6. Reprodução do fluxo CNN descrito no artigo de C. K. L. Caoile.
7. Implementação da comparação com o código MATLAB do File Exchange.
8. Integração do Automatic Component Analyzer e do ResistorPrediction no benchmark.
9. Preparação de manifestações, dataset geral e exports YOLO.
10. Treinos na RTX 5060 remota.
11. Implementação de detector híbrido para várias peças por imagem.
12. Implementação de modo especializado para fitas com resistores repetidos.
13. Implementação de vídeo, tracking por IoU e fusão temporal.
14. Testes unitários, benchmark de robustez e comparação com MATLAB.
15. Produção da proposta acadêmica em PDF, incluindo capa, metodologia, dataset e cronograma.

## 3. Fundamentação técnica

A arquitetura atual é híbrida porque combina as ideias das fontes mais úteis:

| Fonte | Ideia aproveitada | Situação |
|---|---|---|
| C. K. L. Caoile, Automatic Resistor Value Detection Using Computer Vision and Machine Learning, JETSUR, 2026, DOI 10.5281/zenodo.20699559 | ROI, classificação CNN do valor e proposta YOLO--CNN | Reproduzida e ampliada para múltiplos resistores |
| Registro Zenodo 20699559 | Artigo-base e descrição experimental | PDF disponível; o registro não contém dataset anexado |
| MATLAB File Exchange, Resistor Color Code Value Detector From Image | Segmentação, morfologia, componentes conectados e leitura das faixas | Comparado com MATLAB nativo e porta Python |
| Automatic Component Analyzer | Segmentação HSV, ajuste de limiares e leitura de bandas | Código baixado e benchmark executado |
| ResistorPrediction | Extração de perfis, pontos críticos e classificação de cores | Reprodução literal alcançada no conjunto original |
| HiRes | Detecção, máscara, alinhamento, projeção no eixo e validação E24 | Estudado e incorporado como referência; não reproduzido integralmente |
| Datasets Roboflow e Kaggle | Treino de detecção, segmentação e classificação | Baixados, normalizados e/ou usados em benchmarks |

O artigo de Caoile é a referência principal da metodologia. A implementação deste projeto mantém a ideia de separar localização e leitura, mas substitui a classificação de imagem inteira por um pipeline capaz de localizar vários componentes, alinhar cada recorte e validar o valor por regras físicas do código de cores.

## 4. Datasets e materiais

### 4.1 Dataset principal

O dataset principal escolhido é o export YOLO do Roboflow Samu:

- projeto: electronic-components-d6uul;
- diretório: data/sources/roboflow-electronic-components;
- export usado: YOLO;
- 9.850 imagens no pacote exportado;
- referência principal para detecção e contagem.

Ele não deve ser tratado automaticamente como dataset de valor em ohms: a maior parte dos exports de componentes não informa as faixas nem o valor nominal. Por isso, os datasets de detecção e os datasets de valor têm funções diferentes.

### 4.2 Dataset geral agregado

O manifesto geral fica em data/derived/general-dataset:

| Item | Quantidade |
|---|---:|
| Registros totais | 99.242 |
| Objetos compatíveis com detecção | 14.833 |
| Classificação de componente | 25.464 |
| Classificação de valor | 120 |
| Detecção legada | 5.490 |
| Segmentação com máscara | 128 |
| Recortes com valor | 4.257 |
| Imagens sem anotação compatível | 48.950 |

Os arquivos principais são:

- manifest.jsonl: todos os registros descobertos;
- object-detection-manifest.jsonl: objetos compatíveis com detecção;
- value-manifest.jsonl: exemplos com valor ou faixas;
- yolo/: export normalizado de uma classe, resistor;
- summary.json: contagens e fontes da montagem.

Imagens sem anotação compatível foram preservadas, mas não são transformadas artificialmente em negativos supervisionados.

### 4.3 Fontes baixadas

Foram obtidos e organizados:

- Kaggle Eralp Ozcan: 2.881 imagens JPG;
- Kaggle Alptugaydin: imagens e train.json com polígonos VIA;
- Kaggle Barrettotte: 1.270 imagens, 4-band.json e 5-band.json;
- MATLAB File Exchange: pacote completo com 41 arquivos;
- Automatic Component Analyzer: clone Git;
- ResistorPrediction: clone Git com imagens, labels e notebooks;
- Roboflow resistordataset: 135 imagens exportadas;
- Roboflow electronic-ezdmf: 239 imagens exportadas;
- Roboflow electronic-components-d6uul: 9.850 imagens exportadas;
- Roboflow all-components: 35.949 itens no pacote;
- fontes adicionais Kaggle, GitHub e Mendeley listadas em NOTES.md;
- artigo JETSUR/Caoile copiado para data/sources/zenodo/Caoile-paper.pdf.

### 4.4 Fontes incompletas ou não disponíveis

- O Zenodo 20699559 contém o artigo, mas não disponibiliza o dataset. O artigo descreve 1.270 imagens e 89 valores; isso coincide fortemente com Barrettotte, mas a identidade não foi confirmada documentalmente.
- O notebook Kaggle skender85/resistor-evaluationv5 não foi baixado: os endpoints públicos testados retornaram 404 e a API exige credencial/export válido.
- O DirençNet/ResistorNet do Scribd foi mantido somente como página HTML. O documento completo não foi baixado porque a plataforma restringe o conteúdo.
- Os datasets Roboflow de componentes normalmente não têm valor em ohms ou faixas; servem para detecção, não para validar leitura elétrica.

O inventário fonte por fonte está em benchmarks/SOURCES_STATUS.md e benchmarks/SOURCES_AND_APPROACHES.md.

## 5. Arquitetura implementada

O fluxo recomendado é:

1. Aquisição de imagem ou vídeo.
2. Correção de iluminação, redução de ruído e normalização.
3. Detecção dos componentes por YOLO ou método clássico.
4. Filtragem geométrica e eliminação de duplicatas.
5. Recorte e alinhamento pelo eixo do resistor.
6. Análise das faixas em RGB, HSV e CIELAB.
7. Decodificação de 4 ou 5 faixas para ohms e tolerância.
8. Validação por geometria, série E24 e consenso entre métodos/frames.
9. Saída anotada e JSON por componente.

Principais módulos:

- src/resistor_detector/cli.py: comandos image e video;
- src/resistor_detector/detection.py: candidatos, contornos e amostragem;
- src/resistor_detector/advanced.py: segmentação e análise avançada;
- src/resistor_detector/colors.py: mapas e classificação de cores;
- src/resistor_detector/band_decoder.py: decodificação e treino auxiliar;
- src/resistor_detector/automatic_component_analyzer.py: adaptação do projeto externo;
- src/resistor_detector/caoile_cnn.py: reprodução da CNN do artigo;
- src/resistor_detector/matlab_compat.py: equivalente Python do fluxo MATLAB;
- src/resistor_detector/tracking.py: tracking por IoU;
- src/resistor_detector/benchmark.py: métricas, matching e estratégias;
- src/resistor_detector/pipeline.py: fluxo de alto nível.

Ferramentas principais:

- tools/combined_multi_resistor.py: detector híbrido e modo fita;
- tools/build_general_dataset.py: montagem do dataset geral;
- tools/run_benchmark_matrix.py: matriz de métodos e datasets;
- tools/train_yolo_baseline.py: treino YOLO de detecção;
- tools/train_yolo_segmentation.py: treino YOLO de segmentação;
- tools/train_unet_segmentation.py: Tiny U-Net para máscaras;
- tools/train_caoile_imagefolder.py: treino CNN em ImageFolder;
- tools/benchmark_matlab_comparison.py: comparação com o MATLAB;
- tools/benchmark_automatic_component_analyzer.py: comparação com o Analyzer;
- tools/benchmark_resistorprediction_original.py: reprodução do ResistorPrediction;
- tools/benchmark_video.py: avaliação temporal;
- tools/benchmark_robustness.py: brilho, contraste, blur e JPEG;
- tools/evaluate_general_detector.py e evaluate_hires_pipeline.py: avaliações especializadas.

## 6. Resultados registrados

Os valores abaixo vêm de benchmarks/EXECUTION_REPORT.md. Resultados de datasets diferentes não devem ser comparados sem considerar o protocolo.

### 6.1 Melhor detector de objetos

No dataset geral, o treino YOLO agregado alcançou:

| Métrica | Resultado |
|---|---:|
| Precisão | 0,9114 |
| Recall | 0,9318 |
| F1 em IoU 0,5 | 0,9215 |
| mAP50 | 0,9361 |
| mAP50-95 | 0,7511 |
| MAE de contagem | 0,0790 |

O checkpoint final fica em data/results/general-yolo-5060/train/weights/best.pt no host de treino.

No benchmark anterior do Roboflow electronic-components, YOLO11n com 30 épocas obteve mAP50 0,931, mAP50-95 0,812, F1 0,896 e aproximadamente 135,5 FPS.

### 6.2 Vários resistores na mesma imagem

Na comparação com as 16 imagens do pacote MATLAB:

| Métrica | Resultado |
|---|---:|
| Contagem exata por imagem | 16/16, 100% |
| Resistores de referência | 39 |
| Resistores decodificados | 39/39, 100% |
| Valores corretos sem considerar ordem | 37/39, 94,87% |
| Multiconjunto de valores exato | 14/16, 87,50% |
| Sequência exatamente igual à saída MATLAB | 5/16, 31,25% |

A diferença de sequência ocorreu principalmente pela ordem de enumeração dos componentes, não pela ausência dos mesmos valores. Dois casos ainda tiveram divergência de valor, como 4100/4700 ohms.

### 6.3 Fotos do usuário

No modo especializado para fita:

| Imagem | Contagem | Leitura |
|---|---:|---|
| IMG_20260920_232026.jpg | 14 | 470 ohms em 14/14 |
| IMG_20260920_232042.jpg | 14 | 470 ohms em 14/14 |
| IMG_20260920_232046.jpg | 15 | 470 ohms em 15/15 |
| IMG_20260920_232100.jpg | 14 | 470 ohms em 14/14 |
| Total | 57 | 57/57, 100% |

Esse resultado é específico para o cenário controlado das fitas. Ele não prova desempenho equivalente em fotografias arbitrárias.

### 6.4 Comparações externas

- ResistorPrediction original: 54/90 previsões aceitáveis, 60%; 36/90 perfeitas, 40%.
- MATLAB: contagem concordante em 16/16 imagens; sequência concordante em 5/16; mesmo multiconjunto em 14/16.
- Automatic Component Analyzer: configuração padrão acertou 0/16 contagens; melhor varredura acertou 1/16. O código é útil como referência de segmentação HSV, mas não generalizou para as fotos MATLAB.
- Vídeo sintético com tracking: F1 0,7504, precisão 0,6115, recall 0,9708 e 136,5 FPS.
- Robustez: no conjunto avaliado, F1 0,8956 na imagem original, 0,8771 com brilho reduzido, 0,7702 com brilho aumentado, 0,8730 com baixo contraste, 0,8842 com blur e 0,8710 com JPEG qualidade 35.

### 6.5 Conclusão experimental

O melhor caminho operacional atual é:

detector YOLO para localizar -> alinhamento/normalização do recorte -> decoder de faixas e regras físicas -> consenso entre recortes ou frames.

O processamento clássico isolado e a CNN que classifica a imagem inteira têm valor histórico e diagnóstico, mas não são suficientes para a meta de vários resistores em cenas variadas.

## 7. Como usar no Windows

Abra o PowerShell na pasta do repositório:

    cd "C:\caminho\para\detector-resistores"
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"

Para usar YOLO:

    python -m pip install -e ".[dev,detector]"

Para reproduzir a CNN de Caoile:

    python -m pip install -e ".[dev,cnn]"

Imagem simples:

    resistor-detector image caminho\para\imagem.jpg --band-count 4 --output outputs\imagem-anotada.jpg

Vídeo:

    resistor-detector video caminho\para\video.mp4 --every-n-frames 5 --output outputs\video-anotado.mp4

Modo multi-resistor/fita:

    $env:PYTHONPATH="src"
    python tools/combined_multi_resistor.py --input data\inputs\user-images --output-dir data\results\local-strip --device cpu --imgsz 1280 --strip-mode --reference ""

No Windows, use --device 0 se houver uma GPU CUDA configurada. Caso contrário, use --device cpu.

## 8. Como usar no Linux e no pedrogpu

Conexão:

    ssh pedro@192.168.1.57
    cd /home/pedro/detector-resistores
    source .venv/bin/activate
    export PYTHONPATH=src

Treino YOLO agregado:

    python tools/train_yolo_baseline.py \
      --data-yaml data/derived/general-dataset/yolo/data.yaml \
      --ground-truth data/derived/general-dataset/object-detection-manifest.jsonl \
      --output-dir data/results/general-yolo-5060 \
      --epochs 15 --imgsz 640 --batch 16 --device 0

Treino de segmentação:

    python tools/train_unet_segmentation.py \
      --annotation data/sources/kaggle-alptugaydin/extracted/train.json \
      --image-dir data/sources/kaggle-alptugaydin/extracted/train \
      --output data/results/unet-alptugaydin-5060/model.pt \
      --metrics data/results/unet-alptugaydin-5060/metrics.json

Benchmark integrado:

    python tools/combined_multi_resistor.py --device 0

Matriz de métodos:

    python tools/run_benchmark_matrix.py \
      --manifest data/manifests/electronic-components-resistor-detection.jsonl \
      --strategies classical-lab-pca segmentation-rule hybrid-4 hybrid-5 hybrid-auto \
      --resume

O treino pesado foi executado no pedrogpu. O código mantém caminhos relativos e deve continuar executável localmente, mas os datasets e checkpoints grandes podem precisar ser copiados separadamente.

## 9. Testes e verificação

Executar testes unitários:

    python -m pytest -q

O último resultado registrado no Linux remoto foi **14 testes aprovados**. Os testes cobrem:

- mapas de cores;
- decodificação de 4 faixas;
- métricas e matching do benchmark;
- compatibilidade MATLAB;
- segmentação e componentes;
- adaptação do Automatic Component Analyzer.

Para reproduzir a comparação MATLAB:

    python tools/benchmark_matlab_comparison.py \
      --image-dir data/sources/mathworks/extracted/ResistorColorCode/Resistor \
      --output-dir data/results/matlab-comparison

Com MATLAB e Image Processing Toolbox instalados, o harness usa as funções nativas. Sem MATLAB, os shims em tools/matlab_compat_toolbox funcionam como fallback de portabilidade.

## 10. O que ainda falta

### Prioridade alta

1. Criar um conjunto independente de teste com imagens reais do usuário, separado do treino, e anotar caixas, faixas e valores.
2. Obter vídeo real anotado. O benchmark atual de vídeo é sintético.
3. Ampliar as anotações de valor/faixas: os datasets de detecção não são suficientes para validar ohms.
4. Resolver as duas divergências de valor observadas na comparação MATLAB e padronizar a ordenação dos componentes.
5. Copiar para a máquina local os checkpoints e JSON finais que ficaram no pedrogpu, ou disponibilizar um artefato de release.
6. Avaliar o detector agregado em uma divisão externa, evitando que imagens visualmente próximas apareçam em treino e teste.

### Prioridade média

1. Reproduzir integralmente o HiRes, especialmente segmentação de bandas e validação E24.
2. Melhorar o alinhamento de componentes inclinados e parcialmente ocluídos.
3. Calibrar confiança e retornar “ilegível” quando as faixas estiverem ocultas, em vez de forçar um valor.
4. Melhorar a estabilidade temporal para evitar troca de IDs e contagem duplicada.
5. Transformar o modo fita em uma configuração automática detectada pelo sistema.
6. Documentar licenças e condições de uso de cada dataset antes de distribuição pública.

### Infraestrutura

1. Fazer o primeiro commit do repositório e configurar um remoto GitHub/GitLab, se desejado.
2. Não versionar datasets brutos, credenciais, resultados pesados nem checkpoints grandes; usar releases, armazenamento externo ou Git LFS.
3. Adicionar CI para instalar o pacote e executar pytest em Windows e Linux.
4. Criar um arquivo de configuração para caminhos, dispositivo, limiar, tamanho de imagem e diretórios de saída.

## 11. Organização do repositório

- README.md: início rápido e comandos principais.
- NOTES.md: histórico de pesquisa, fontes adicionais e ideias futuras.
- benchmarks/SOURCES_STATUS.md: status de download de cada fonte.
- benchmarks/SOURCES_AND_APPROACHES.md: análise técnica fonte por fonte.
- benchmarks/EXECUTION_REPORT.md: resultados completos dos experimentos.
- benchmarks/GENERAL_DATASET.md: manifesto e regras do dataset agregado.
- benchmarks/CAOILE_REPRO.md: reprodução do artigo de Caoile.
- src/resistor_detector/: pacote principal.
- tools/: scripts de preparação, treino e benchmark.
- tests/: testes unitários.
- data/sources/: datasets e códigos baixados; ignorado pelo Git.
- data/results/: resultados e checkpoints; ignorado pelo Git.
- data/derived/: manifests e exports derivados; avaliar tamanho antes de versionar.
- runs/: saídas de treino; não deve ser enviado ao Git.
- Proposta_Processamento_de_Sinais_2_Deteccao_de_Resistores_Pedro_Bastos.pdf: proposta acadêmica final.

## 12. Referências principais

- Caoile, C. K. L. Automatic Resistor Value Detection Using Computer Vision and Machine Learning. Journal of Engineering and Technology for Sustainability and Resilience, vol. 1, n. 1, 2026. DOI: 10.5281/zenodo.20699559.
- MATLAB File Exchange. Resistor Color Code Value Detector From Image.
- bytesizedengineering. automatic-component-analyzer.
- nichyjt. ResistorPrediction.
- HiRes491. HiRes: A Hierarchical Cascaded Method for Resistor Value Identification.
- Datasets e links adicionais: NOTES.md e benchmarks/SOURCES_AND_APPROACHES.md.

## 13. Estado do Git

O diretório já é um repositório Git local, mas inicialmente estava sem commits. Os datasets brutos, resultados e ambientes virtuais estão protegidos pelo .gitignore. Depois de revisar esta documentação, deve ser criado o primeiro commit contendo código, testes, scripts e documentação, sem incluir os arquivos grandes ignorados.
