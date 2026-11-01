# Resultados da implementação da proposta

Os experimentos abaixo foram executados em `pedrogpu`, com RTX 5060. As tabelas
se referem aos notebooks desta branch; resultados históricos de outros
protocolos permanecem no relatório de benchmark anterior.
Ambiente: Python 3.12, PyTorch 2.7.0 com CUDA 12.8, Ultralytics 8.3.92,
OpenCV 5.0.0, NumPy 2.1.1, pandas 3.0.6 e Matplotlib 3.11.2.

## Dados e divisão

O export Samu possui 9.850 imagens. As anotações filtradas contêm 2.794 imagens
com resistores e 3.947 caixas. O agrupamento de nomes de origem e cópias
idênticas resultou nesta divisão:

| Partição | Imagens | Imagens com resistor | Caixas |
| --- | ---: | ---: | ---: |
| Treino | 6.894 | 1.884 | 2.696 |
| Validação | 1.478 | 421 | 647 |
| Teste | 1.478 | 489 | 604 |

O agrupamento reduz vazamento de variações conhecidas, mas o export não possui
identidade das cenas ou dos componentes; a independência física ainda precisa
ser verificada manualmente.

## Métodos clássicos

Na amostra exploratória de 200 imagens da validação, o melhor resultado de
contagem exata das nove combinações testadas foi **45,5%**, com CIELAB sem
correção. Esse percentual inclui imagens negativas. Nesse mesmo experimento,
precisão e recall de caixas em IoU 0,5 foram **9,2%** e **25,9%**. As demais
combinações e tempos estão no notebook 3 e nos CSVs produzidos por ele.

## YOLO treinado no Samu

O YOLO11n foi treinado por 30 épocas na divisão acima. No teste de 1.478
imagens, o avaliador Ultralytics calculou **mAP50 de 0,8830** e **mAP50–95 de
0,7979**. Com confiança fixa de 0,25, o benchmark por imagem obteve:

| Medida | Clássico | YOLO |
| --- | ---: | ---: |
| Contagem exata, todas as imagens | 33,8% | 95,9% |
| Contagem exata, 489 imagens positivas | 44,2% | 90,4% |
| Erro absoluto médio de contagem | 1,097 | 0,125 |
| Precisão de caixas, IoU ≥ 0,5 | 15,5% | 76,7% |
| Recall de caixas, IoU ≥ 0,5 | 43,5% | 83,9% |
| Tempo médio por imagem | 1,25 ms | 6,40 ms |

Os tempos são da etapa de detecção após aquecimento e não incluem a leitura
das faixas. O índice de 95,9% inclui 989 imagens sem resistor e, portanto, não
atinge a meta de 95% quando considerados somente os casos positivos. A escolha
de outro limiar no notebook 5 usa exclusivamente a validação.

O limiar escolhido na validação foi **0,50**. Aplicado ao teste reservado,
resultou em contagem exata de **96,8%** no conjunto completo e **90,8%** nas
imagens positivas, erro absoluto médio de **0,091**, precisão de caixas de
**99,2%**, recall de **77,6%** e F1 de **87,1%**. O tempo médio de detecção
foi 7,32 ms por imagem; os tempos variam conforme a carga da máquina e não
incluem leitura de faixas. A seleção do limiar e a avaliação usam IoU ≥ 0,5.

## Leitura de valores

No Barrettotte, 200 imagens de grupos reservados foram avaliadas com a imagem
inteira como entrada. A cobertura foi 5% para o consenso de perfis, 8% para a
porta MATLAB e 21,5% para o Automatic Component Analyzer; nenhuma das
leituras emitidas acertou o valor nominal nesse teste. Essas imagens não têm
caixas de corpo confiáveis no manifesto; portanto, o resultado evidencia
também as limitações de localização e recorte, não apenas as cores.

Em uma amostra exploratória de 200 imagens Eralp, o detector geral antigo com
o segmentador HiRes emitiu valor em 34,5% das imagens e acertou 10,5% do total
(21 imagens). O histórico de treino dos pesos de terceiros não foi auditado e
pode haver sobreposição com esses dados. O resultado não é uma estimativa
independente de generalização.

Na mesma amostra, o detector novo treinado no Samu, com limiar 0,25 e o
segmentador HiRes, emitiu apenas 6 leituras (3% de cobertura) e não acertou
nenhum valor nominal. Esse experimento mostra uma transferência ruim para
essas imagens; não é uma avaliação independente do checkpoint HiRes.

Um novo segmentador foi treinado no notebook 7 com **487 recortes de treino,
94 de validação e 112 de teste**, agrupados por valor/tolerância. Após 15
épocas, o melhor checkpoint teve IoU médio de faixas de **0,439** no teste.
Ele emitiu leituras em 106 dos 112 recortes e acertou **25/112 valores
(22,3%)**, ou **25/106 (23,6%)** entre os valores emitidos. A tolerância foi
identificada corretamente em **61,6%** dos recortes, e **25,9%** das leituras
ficaram dentro da tolerância nominal; ambas as métricas incluem as recusas.
A IoU foi baixa para algumas cores, incluindo amarelo (0,021). Não havia pixels
violetas anotados nesse teste; o IoU zero para essa classe reflete previsões
indevidas, não uma medida de recall de violeta. O teste usa
recortes previamente anotados: não é comparável diretamente ao teste com
imagens completas acima nem valida o sistema de ponta a ponta. A independência
de identidade física também permanece desconhecida.

Em seguida, foi treinada uma segunda versão por **40 épocas**, com o mesmo
protocolo de grupos. O IoU médio no teste foi **0,561**; a cobertura de leitura,
**108/112 (96,4%)**; o valor exato, **67/112 (59,8%)**; e a tolerância exata,
**84,8%**. Os valores dentro da tolerância nominal corresponderam a **63,4%**
dos recortes. Entre as 108 leituras emitidas, 67 valores estavam corretos
(62,0%). Como a decisão de ampliar o treinamento ocorreu depois de observar o
primeiro teste, esta comparação é **exploratória** e exige novo teste independente.

## Exemplo integrado com múltiplos resistores

Na imagem `res13clc.jpg` do pacote MathWorks, a anotação por inspeção visual
contém quatro resistores: 10 kΩ, 220 Ω, 1 kΩ e 220 Ω. Usando o detector geral
anterior a 960 pixels, o segmentador HiRes e o fallback compatível com MATLAB,
o notebook 4 localizou os quatro e apresentou os quatro valores corretos.
Identificou também a tolerância de dois; nos outros dois, deixou a tolerância
em aberto para revisão. As caixas foram associadas à referência com IoU ≥ 0,5.

| Resistor | Valor de referência | Valor apresentado | Tolerância apresentada |
| --- | ---: | ---: | ---: |
| R1 | 10 kΩ | 10 kΩ | Revisar |
| R2 | 220 Ω | 220 Ω | ±5% |
| R3 | 1 kΩ | 1 kΩ | Revisar |
| R4 | 220 Ω | 220 Ω | ±5% |

Este é um teste de funcionamento em uma única imagem, não comprova a meta de
95% e não substitui teste independente com cenas e peças diferentes. A
anotação não é oficial e deve ser conferida contra as peças ou medição.
No mesmo exemplo, o detector recém-treinado apenas no Samu, com resolução
640 e confiança 0,25, não detectou os quatro resistores. Assim, os resultados
do exemplo com o detector geral antigo não podem ser atribuídos ao novo modelo.
Quando o segmentador recém-treinado substituiu o HiRes antigo, mantendo o
detector geral, os quatro foram localizados; dois valores foram emitidos
corretamente e os outros dois ficaram em revisão por discordância entre os
métodos. A divergência não foi convertida em um valor arbitrário.
Com o segmentador de 40 épocas, o mesmo detector localizou quatro peças, mas
somente um valor foi emitido; os outros três ficaram em revisão por discordância.

## Pendências de validação

- Obter e conferir anotações conjuntas de caixa, sequência, orientação,
  resistência e tolerância em fotografias próprias com resistores separados.
- Medir a leitura do pipeline completo nessas imagens, inclusive as recusas;
  nenhuma das métricas acima demonstra desempenho geral em múltiplos objetos.
