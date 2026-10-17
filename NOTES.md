https://zenodo.org/records/20699559

https://www.kaggle.com/datasets/eralpozcan/resistor-dataset

https://www.kaggle.com/datasets/alptugaydin/resistor-images-dataset-for-segmentation

https://www.kaggle.com/datasets/barrettotte/resistors

https://www.kaggle.com/code/skender85/resistor-evaluationv5/notebook

https://www.mathworks.com/matlabcentral/fileexchange/73099-resistor-color-code-value-detector-from-image

https://github.com/bytesizedengineering/automatic-component-analyzer

https://universe.roboflow.com/harish-madhavan/resistordataset

https://universe.roboflow.com/abc-qqksz/electronic-ezdmf

https://universe.roboflow.com/samu-drioq/electronic-components-d6uul


https://universe.roboflow.com/ned-university-of-engineering-and-technology-5f8dq/all-components

https://github.com/nichyjt/ResistorPrediction

C:\Users\pedro\OneDrive - cefet-rj.br\Documentos\Faculdade\processamento de sinais 2\detector-resistores\JETSUR_01_01_009_(C.K.L. Caoile).pdf


## Fontes adicionais pesquisadas — 2026-09-20

As fontes abaixo foram encontradas depois do levantamento inicial. Elas ainda são **candidatas de pesquisa**; não foram incorporadas automaticamente ao treinamento. O status indica se a fonte foi apenas analisada ou se já foi reproduzida no projeto.

### Datasets de resistores/componentes

1. **Resistor Detection from Circuit Drawings** — [Kaggle](https://www.kaggle.com/datasets/nyahmet/resistor-detection-from-circuit-drawings)
   - 266 imagens de circuitos, incluindo desenhos digitais e feitos à mão, com anotações YOLO.
   - Serve para detectar o símbolo de resistor em diagramas; não contém fotografias de resistores nem faixas de cor.
   - Uso recomendado: experimento separado de leitura de esquemas, não treinamento direto do leitor de valor.
   - Status: candidato; não baixado.

2. **Electronic Component Recognizer** — [Kaggle](https://www.kaggle.com/datasets/mdfaisalahmed025/electronic-component-recognizer)
   - A descrição informa 3.661 imagens rotuladas em 20 classes, incluindo `Resistor`.
   - Útil para pré-treinamento/filtragem de componente, mas a classe não implica anotação de faixas ou valor ôhmico.
   - Status: candidato; não baixado.

3. **Electronic components and devices** — [Kaggle](https://www.kaggle.com/datasets/aryaminus/electronic-components)
   - Coleção com mais de 11 mil imagens, organizada em 36 diretórios/classe, incluindo componentes como resistores.
   - Pode ampliar negativos e variação de aparência para classificação/detecção de componentes.
   - Status: candidato; não baixado.

4. **resistance-detection2 / resistor_component_detection_yolo12** — [GitHub](https://github.com/share2code99/resistor_component_detection_yolo12)
   - A documentação descreve 564 imagens em formato YOLOv8, uma classe `resistor`, resolução normalizada para 1024x1024 e divisão treino/validação/teste.
   - O foco é detectar resistor em placas; não há garantia de rótulo confiável de valor/faixas. A página também menciona CC BY 4.0 para o dataset, mas a licença deve ser conferida no download original.
   - O repositório descreve uma variante YOLOv12 com A2C2f/DFFN/DYT e busca por melhor desempenho em objetos pequenos.
   - Status: candidato; não baixado nem validado independentemente.

5. **ElectroCom61** — [Mendeley Data](https://data.mendeley.com/datasets/6scy6h8sjz/2) e [repositório/código](https://github.com/faiyazabdullah/ElectroCom61)
   - Versão 2 publicada em 2024: 2.121 imagens anotadas, 61 classes, iluminação/fundos/ângulos variados e divisão 70/20/10.
   - A licença informada na página do Mendeley é CC BY 4.0. Serve para detecção de componentes reais e robustez de captura, não para decodificar faixas.
   - Status: candidato; não baixado.

6. **AITEE Electrical Circuit Dataset** — [GitHub](https://github.com/CKnievel/aitee-dataset)
   - 831 diagramas de circuitos, 11 classes, anotações YOLOv8 e licença Apache-2.0; inclui a classe `R` para resistor e identificadores como `R1`.
   - Inclui pipeline de detecção YOLOv8, Canny/contornos e geração de netlist.
   - Útil para uma futura modalidade de entrada “esquema elétrico”; não é dataset de foto nem de cor de resistor.
   - Status: candidato; não baixado.

7. **PCB Component Detection YOLOv8** — [GitHub](https://github.com/TalhaAlvi1/pcb-component-detection-yolov8)
   - Dataset de PCB com 23 classes, 548/80/44 imagens treino/val/teste e resolução 1280x1280.
   - O relatório publicado mostra mAP@0.5 global de 42,6%, mas apenas 8,9% para `Resistor` e 5,5% para `Resistor Network`; é uma evidência útil de que resistores pequenos exigem alta resolução e dados específicos.
   - Licença do repositório: MIT.
   - Status: referência de benchmark; não baixado.

8. **Yolov8_resistor_detection** — [GitHub](https://github.com/Vinit5893/Yolov8_resistor_detection)
   - O repositório informa 4.422 imagens/labels de treino e 486 imagens/labels de validação, obtidos de um projeto Roboflow.
   - Serve para detecção de resistor como objeto, mas a própria documentação diz que os dados foram baixados de outra fonte e não anotados pelo autor; a procedência/licença precisa ser verificada antes de uso.
   - Status: candidato; não baixado.

9. **Power Equipment Image Dataset** — [GitHub](https://github.com/xiongsiheng/Power-equipment-image-dataset)
   - Dataset de equipamentos de potência com subset de detecção e classe `resistor`, além de outras classes de subestação.
   - É relevante apenas como fonte adicional de contexto de placas/equipamentos; não é apropriado como fonte primária para resistor axial com faixas coloridas.
   - Status: candidato secundário; não baixado.

### Implementações e métodos adicionais

1. **HiRes: A Hierarchical Cascaded Method for Resistor Value Identification** — [artigo](https://arxiv.org/abs/2606.30179) e [código](https://github.com/HiRes491/HiRes)
   - Pipeline em cascata: YOLOv8n para detecção, UNet++ com EfficientNet-B2 para segmentação, projeção ao longo do eixo do resistor, separação de faixas preservando os espaços e validação pela série E24.
   - O artigo reporta mAP50 de detecção 0,9906, mIoU de segmentação 0,8444 e 85,8% de identificação end-to-end no conjunto do trabalho. Esses números ainda não foram reproduzidos neste projeto.
   - É a referência mais próxima da arquitetura híbrida desejada: aprendizado para localizar/segmentar e regras físicas para decodificar.
   - Status: analisado; código/dataset não incorporados.

2. **CNN para detecção/localização de resistores de faixas em PCB** — [artigo JEIT](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT190608?viewType=HTML)
   - Usa CNN encoder-decoder com cross-entropy ponderada para segmentar o resistor em cenas de PCB.
   - Depois calcula o retângulo mínimo, aplica transformação afim para alinhar o resistor verticalmente e localiza as faixas com template matching Gaussiano.
   - Foi avaliado em 1.270 imagens de PCB e comparado a contornos geométricos/template matching tradicionais.
   - Adaptação recomendada: trocar o template rígido por máscara/instância segmentada e manter a etapa de alinhamento afim.
   - Status: método analisado; não reproduzido.

3. **Reading resistor based on image processing** — [publicação/PDF](https://www.researchgate.net/publication/308734249_Reading_resistor_based_on_image_processing)
   - Método clássico para 4 e 5 faixas: restringe a região pelo eixo do resistor, estima a cor do corpo pelos cantos, subtrai o corpo, limiariza as mudanças e extrai as faixas.
   - Usa características HSL e KNN (K=1), com bases de treinamento separadas para resistores de 4 e 5 faixas para reduzir confusão entre dourado/prata.
   - Útil como baseline interpretável e fallback quando a segmentação neural falhar.
   - Status: método analisado; não reproduzido neste ciclo.

4. **ResistorScanner** — [GitHub](https://github.com/thegouger/ResistorScanner)
   - Implementação Android/OpenCV: mascara cores, encontra contornos, associa cada cor ao centróide e ordena os códigos pela coordenada x.
   - Assume a faixa de tolerância à direita e funciona melhor com fundo claro; o próprio projeto alerta para diferenças de white balance entre câmeras.
   - Adaptação recomendada: usar orientação estimada automaticamente, calibração de balanço de branco e confidência por faixa.
   - Status: método analisado; não reproduzido.

5. **Real-Time Resistor Color Code Recognition using Image Processing in Mobile Devices** — [publicação](https://www.researchgate.net/publication/325967064_Real-Time_Resistor_Color_Code_Recognition_using_Image_Processing_in_Mobile_Devices)
   - Referência voltada a análise de vídeo móvel e reconhecimento de código de cores em tempo real, incluindo o problema de orientação e resistores de 5 faixas.
   - Relevante para o modo vídeo do projeto: estabilização temporal, seleção de frames nítidos e agregação de leituras, em vez de decidir a partir de um único frame.
   - Status: referência metodológica; não reproduzido.

6. **Automated Resistor Classification** — [relatório ETH](https://pub.tik.ee.ethz.ch/students/2014-HS/GA-2014-02.pdf)
   - Explora segmentação baseada em saturação/HSV e representação Lab para separar o corpo do resistor e classificar sua aparência.
   - Pode fornecer features auxiliares para detectar corpo/fundo e estimar qualidade de iluminação antes da leitura das faixas.
   - Status: referência metodológica; não reproduzido.

### Síntese para a arquitetura do projeto

As novas fontes reforçam a separação em estágios:

1. **Detecção/contagem:** YOLO de alta resolução ou segmentação de instância; datasets de PCB/componentes podem ajudar somente nesta etapa.
2. **Recorte e orientação:** máscara do resistor, retângulo mínimo/OBB e transformação afim para alinhar o eixo.
3. **Separação das faixas:** projeção de cor ao longo do eixo, preservando gaps; combinar máscara neural com HSV/HSL/Lab e, se necessário, template/contorno.
4. **Classificação de cor:** modelo por faixa com RGB normalizado + HSV/Lab, calibração de white balance e ensemble de regras/KNN/SVM/CNN.
5. **Decodificação física:** testar 4/5/6 faixas, tolerância e coeficiente térmico; validar combinação pela tabela de código e série E24/E96 quando aplicável.
6. **Vídeo:** selecionar frames com foco/blur aceitável, rastrear o mesmo resistor e agregar probabilidades/códigos ao longo do tempo.
7. **Confiabilidade:** emitir valor, faixas, tolerância, confiança e motivo de rejeição; não forçar um valor quando a geometria ou a sequência de bandas for inconsistente.

Conclusão de pesquisa: datasets de diagramas e de PCB são bons para a etapa de localizar componentes, mas os datasets com anotações específicas de faixas/valor continuam sendo o gargalo. Para o objetivo principal, a combinação mais promissora é **detector/segmentador aprendido + alinhamento geométrico + classificador de cor calibrado + decodificador físico com validação E24 + fusão temporal no vídeo**.

## Status de download e execução — 2026-09-20

### Fontes baixadas para `pedrogpu`

- Kaggle: `nyahmet/resistor-detection-from-circuit-drawings`, `mdfaisalahmed025/electronic-component-recognizer` e `aryaminus/electronic-components`.
- GitHub: `share2code99/resistor_component_detection_yolo12`, `Vinit5893/Yolov8_resistor_detection`, `CKnievel/aitee-dataset`, `HiRes491/HiRes`, `TalhaAlvi1/pcb-component-detection-yolov8`, `xiongsiheng/Power-equipment-image-dataset` e `faiyazabdullah/ElectroCom61`.
- Mendeley: `ElectroCom61` v2, ZIP validado e extraído em `data/sources/mendeley-electrocom61/extracted`.
- As fontes anteriores (Kaggle, Roboflow, MathWorks, Zenodo, Caoile, Automatic Component Analyzer e ResistorPrediction) permanecem em `data/sources`.

### Fontes parciais ou não utilizáveis como dataset rotulado

- Roboflow `resistordataset`: exportação API validada e extraída; 135 imagens e 132 objetos de resistor entraram no manifesto agregado.
- Scribd/DirençNet: mantido apenas como página HTML; a fonte foi explicitamente retirada do escopo do dataset.
- TalhaAlvi1 e `electrocom61` no GitHub: código/relatório; os dados utilizáveis vieram da fonte Mendeley.
- Share2Code: clone contém documentação e imagens de exemplo, mas não o dataset completo rotulado.
- Power Equipment: baixado e indexado como imagens sem anotação de resistor compatível; não entrou no treino supervisionado.

### Dataset geral

O manifesto agregado em `data/derived/general-dataset` contém 99.242 registros de todas as fontes disponíveis: 14.833 objetos para detecção, 25.464 exemplos de classificação de componente, 4.257 recortes de valor, 128 máscaras, 5.490 exemplos de detecção legados e 48.950 imagens sem anotação. O treino supervisionado usa somente os registros com rótulo compatível; imagens sem rótulo não são convertidas artificialmente em negativos.

### Execução

O treino YOLO geral e a sequência pós-treino foram concluídos na RTX 5060 de `pedrogpu`: detector agregado com 15 épocas, Tiny U-Net com 30 épocas e seis treinos YOLO específicos com 5 épocas. Os resultados finais estão em `benchmarks/EXECUTION_REPORT.md` e os checkpoints/JSON em `data/results/` no host remoto.

Foi acrescentado o teste integrado de múltiplos resistores `tools/combined_multi_resistor.py`. Ele combina detector agregado, YOLO-seg por instância e segmentação/decodificação HiRes; quando a segmentação de bandas HiRes não generaliza para as fotos MATLAB, usa o decoder MATLAB-compatível por resistor. Na validação de 16 imagens: contagem 16/16, 39/39 valores decodificados e 37/39 valores coincidentes com o MATLAB nativo. Saídas em `data/results/combined-multi-resistor-matlab/`.

### Teste das fotos do usuário — modo fita

Para fotos com muitos resistores iguais em fita, foi acrescentado um modo especializado em `tools/combined_multi_resistor.py`:

- inferência YOLO de segmentação em alta resolução (`imgsz=1280`, limiar 0,05);
- filtro geométrico do corpo do resistor e supressão por distância no eixo da fita;
- propostas adicionais pela primeira faixa amarela para recuperar componentes pequenos ou na borda;
- rejeição de caixas sem evidência de corpo amarelo;
- consenso de imagem para crops desfocados, mantendo o valor somente quando a maioria é 470 Ω.

Execução feita na RTX 5060 de `pedrogpu`:

```bash
.venv/bin/python tools/combined_multi_resistor.py \
  --input data/inputs/user-images \
  --output-dir data/results/combined-user-images-strip-v2 \
  --device 0 --imgsz 1280 --strip-mode --reference ''
```

Resultado nas quatro fotos: `14 + 14 + 15 + 14 = 57` resistores; `57/57` decodificados como `470 Ω`; taxa de decodificação `100%`. As imagens anotadas e `summary.json` estão em `data/results/combined-user-images-strip-v2/`.
