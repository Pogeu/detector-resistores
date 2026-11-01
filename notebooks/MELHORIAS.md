# Melhorias dentro da proposta

Esta análise confronta a [proposta aprovada](../Proposta_Processamento_de_Sinais_2_Deteccao_de_Resistores_Pedro_Bastos.pdf)
com os [notebooks](README.md) e os [resultados medidos](RESULTADOS.md). O
objetivo continua sendo contar resistores axiais **separados em imagens
estáticas** e identificar individualmente o valor e a tolerância. Vídeos e
fitas densas permanecem fora do escopo inicial.

## Prioridade 1 — teste conjunto com referências confiáveis

**Problema.** O Samu não tem valores ou tolerâncias, e o teste com 112 recortes
de faixas não mede a localização. O exemplo com quatro resistores usa o
detector geral anterior e uma única imagem anotada por inspeção visual. Ele
não demonstra desempenho do pipeline em cenas novas. O teste do segmentador
de 40 épocas é exploratório, pois foi realizado após observar a versão de 15.

**Melhoria.** Reunir fotografias próprias com um e com vários resistores
separados. Para cada peça, conferir e registrar caixa, sequência de cores,
orientação, valor e tolerância; usar identificação visual e/ou medição
independente para conferir os valores. Revisar as anotações antes de medir.
Separar as imagens por cena **e identidade física das peças**, mantendo na
mesma partição fotos que compartilhem qualquer um desses elementos. O
[notebook 6](06_anotacoes_proprias.ipynb) já valida esse formato e agrupa
cenas e peças identificadas.

Reservar um teste novo antes de fazer ajustes. Reportar contagem exata e erro
absoluto, precisão/recall e mAP das caixas, acertos de faixas, valor e
tolerância, tempo total e proporção de leituras recusadas. Avaliar o mesmo
pipeline completo em todas as imagens, relacionando cada resultado à sua
caixa. Sem essa anotação, não há evidência suficiente de que o sistema atende
à proposta em imagens com vários componentes.

## Prioridade 2 — localização e contagem de resistores pequenos

**Problema.** No teste Samu, com limiar escolhido na validação, a contagem
exata foi 96,8% considerando todas as imagens, mas 90,8% entre as 489 imagens
positivas; o recall de caixas foi 77,6%. A meta de 95% ainda não foi atingida
nas imagens com resistores. Na imagem MathWorks de quatro peças, o YOLO novo
do Samu a 640 pixels não encontrou os componentes, enquanto o detector
geral anterior a 960 pixels os localizou. Isso sugere uma dificuldade de
transferência e escala, mas uma imagem não determina sua causa.

**Melhoria.** Examinar na validação os erros por tamanho da caixa, orientação,
quantidade de peças, fundo e iluminação. Comparar resoluções de inferência e,
se necessário, melhorar as anotações ou complementar o treino com exemplos de
resistores separados e pequenos. O Samu continua sendo a base principal. Para
cada modelo e resolução, escolher os parâmetros **somente na validação** e
registrar o custo em tempo e memória. Depois aplicar a configuração escolhida
ao teste reservado, incluindo imagens positivas e negativas.

O limiar de 0,50 do [notebook 5](05_limiar_e_teste.ipynb) foi escolhido para
um checkpoint e resolução específicos. O [notebook 4](04_segmentacao_faixas.ipynb)
pode carregar esse limiar ao usar outro peso ou resolução; antes de reutilizá-lo,
vincular a calibração ao identificador do modelo, tamanho de entrada e conjunto
de validação, evitando aplicar uma configuração não validada a outro modelo.

## Prioridade 3 — faixas, orientação e tolerância

**Problema.** No teste de recortes anotados do segmentador de 15 épocas, o
valor exato foi 25/112 (22,3%); o experimento de 40 épocas obteve 67/112
(59,8%), mas é exploratório pelo uso repetido do teste. A leitura das regras
clássicas em imagens inteiras teve desempenho baixo e inclui o efeito de
localização e recorte. Esses protocolos não são diretamente comparáveis.

**Melhoria.** Avaliar separadamente: qualidade do recorte, alinhamento pelo
eixo, segmentação das faixas, classificação de cada cor, ordem/orientação e
decodificação final. A amostragem clássica em
`src/resistor_detector/detection.py` usa posições igualmente espaçadas;
compará-la com perfis que localizem transições e com máscaras de faixas já
disponíveis. Investigar cores com pouca representação, reflexos, balanço de
branco e variações de fundo; ampliar dados ou anotações apenas quando necessário.

Escolher a orientação e os limiares usando a validação. Medir cobertura e
acurácia **incluindo recusas**, além de acerto entre leituras emitidas, para
evitar melhorar um número apenas recusando casos difíceis. Quando a ordem, a
cor ou a tolerância forem ambíguas, manter o aviso de revisão e não preencher
um valor por suposição. A confiança da caixa do YOLO não é confiança do valor;
calibrar a confiança da leitura em dados anotados antes de interpretá-la como
probabilidade.

## Prioridade 4 — comparação justa e reprodução

**Problema.** No [notebook 3](03_preprocessamento.ipynb), a comparação usa
cinza derivado de RGB, saturação HSV e distância em CIELAB: são sinais e
limiares diferentes, portanto o resultado não isola somente o espaço de cor.
No notebook 4, o teste em lote do Eralp usa o segmentador HiRes, enquanto a
demonstração individual acrescenta o fallback compatível com MATLAB; os dois
resultados não medem o mesmo pipeline. O agrupamento do Samu por nome de
origem e hash reduz vazamentos conhecidos, mas não identifica todas as cenas
ou peças físicas.

**Melhoria.** Comparar iluminação, ruído e espaços de cor com o mesmo conjunto,
método de segmentação, caixas de referência e métrica; registrar as diferenças
de implementação. Executar o híbrido completo e as referências MATLAB e
Automatic Component Analyzer nas mesmas imagens, com o mesmo protocolo e sem
tratar a saída de um método como verdade de referência.

Registrar versão/origem dos dados, divisão, seed, configurações, hash dos
pesos e ambiente em cada execução. Auditar rótulos e possíveis imagens quase
duplicadas entre partições e fontes; quando a origem ou a identidade física não
puder ser estabelecida, declarar essa limitação. Testar a execução dos notebooks
em Windows e Linux e manter os arquivos grandes na pedrogpu, conforme a
[documentação](README.md).

## Ordem sugerida

1. Anotar e reservar o conjunto próprio; revisar as anotações.
2. Ajustar localização e contagem na validação, sobretudo para peças pequenas.
3. Melhorar e calibrar a leitura das faixas nos recortes e integrar as etapas.
4. Executar uma única avaliação final no conjunto reservado e documentar erros,
   recusas e limitações junto das métricas.
