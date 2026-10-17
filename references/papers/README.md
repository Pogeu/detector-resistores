# Papers usados como base

Esta pasta reúne os PDFs científicos e relatórios técnicos que serviram de base para a metodologia do detector de resistores.

## Documentos disponíveis localmente

### Automatic Resistor Value Detection Using Computer Vision and Machine Learning

- Arquivo: JETSUR_01_01_009_(C.K.L. Caoile).pdf
- Autor: Clark Kneil L. Caoile
- Publicação: Journal of Engineering and Technology for Sustainability and Resilience, vol. 1, n. 1, 2026
- DOI/Zenodo: 10.5281/zenodo.20699559
- Papel no projeto: referência principal para ROI, classificação CNN do valor e proposta YOLO--CNN.
- Observação: o registro Zenodo contém o artigo, mas não contém o dataset. O dataset descrito no artigo parece coincidir com Barrettotte, mas a identidade não foi confirmada documentalmente.

### Resistor Color Code or Value Detector from Image

- Arquivo: EEE212_Project_Report_1806064-65.pdf
- Autores: Afsana Islam Keya e Yeaz Mahmud
- Instituição: Bangladesh University of Engineering and Technology
- Papel no projeto: referência técnica adicional para leitura do código de cores a partir de imagem e organização de um detector educacional.

## Referências online complementares

As referências abaixo foram analisadas ou usadas no benchmark, mas não possuem necessariamente um PDF local nesta pasta:

- HiRes: A Hierarchical Cascaded Method for Resistor Value Identification  
  https://arxiv.org/abs/2606.30179
- Código do HiRes  
  https://github.com/HiRes491/HiRes
- MATLAB Resistor Color Code Value Detector From Image  
  https://www.mathworks.com/matlabcentral/fileexchange/73099-resistor-color-code-value-detector-from-image
- Automatic Component Analyzer  
  https://github.com/bytesizedengineering/automatic-component-analyzer
- ResistorPrediction  
  https://github.com/nichyjt/ResistorPrediction
- DirençNet/ResistorNet  
  https://pt.scribd.com/document/381301116/DirencNet-ResistorNet

## Como os papers influenciaram a implementação

O pipeline final combina:

1. localização do componente por detector ou segmentação;
2. recorte e alinhamento pelo eixo do resistor;
3. análise das faixas em RGB, HSV e CIELAB;
4. classificação das cores;
5. decodificação física para ohms e tolerância;
6. validação por geometria, série E24 e consenso temporal.

Essa combinação segue a separação proposta por Caoile entre localização e classificação, mas amplia o escopo para vários resistores na mesma imagem e para vídeos. As abordagens MATLAB, Automatic Component Analyzer e ResistorPrediction foram mantidas como baselines de comparação.
