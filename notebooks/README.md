# Trabalho da proposta — notebooks

O objetivo é contar resistores axiais separados em imagens estáticas e estimar
individualmente as faixas, a resistência e a tolerância. Vídeo e fitas densas
ficam fora desta implementação. Leia a [proposta aprovada](../Proposta_Processamento_de_Sinais_2_Deteccao_de_Resistores_Pedro_Bastos.pdf);
o [LaTeX e suas imagens](../docs/proposta/proposta_resistores.tex) também estão no projeto.
Os resultados medidos e suas limitações estão em [RESULTADOS.md](RESULTADOS.md).
Os resumos numéricos estão em [resultados/](resultados/README.md).
As prioridades para melhorar o trabalho, sem ampliar o escopo, estão em
[MELHORIAS.md](MELHORIAS.md).

1. `01_dados_e_deteccao.ipynb`: inventário, divisão, pré-processamento,
   treinamento YOLO e avaliação de contagem e localização.
2. `02_faixas_e_integracao.ipynb`: alinhamento, leitura de quatro/cinco faixas,
   avaliação de leitura e saída individual por imagem. As implementações
   anteriores de MATLAB e Automatic Component Analyzer são referências de
   comparação, importadas de `src/`.
3. `03_preprocessamento.ipynb`: comparação exploratória de RGB, HSV e CIELAB
   com redução de ruído e correção de iluminação na validação.
4. `04_segmentacao_faixas.ipynb`: teste do segmentador de faixas HiRes já
   estudado, com avaliação exploratória no conjunto complementar Eralp e
   demonstração com vários resistores.
5. `05_limiar_e_teste.ipynb`: escolha do limiar na validação e avaliação no
   teste reservado. Execute-o depois do notebook 1 e antes da demonstração
   integrada para usar o limiar selecionado.
6. `06_anotacoes_proprias.ipynb`: validação de anotações próprias, separação
   por cena e identidade das peças e avaliação conjunta de contagem, caixas,
   valores e tolerâncias quando as previsões anotadas estiverem disponíveis.
7. `07_treino_faixas.ipynb`: treino e teste de um segmentador de faixas a
   partir de máscaras Barrettotte/HiRes, com grupos separados por valor.

## Ambiente

Em Windows ou Linux, crie um ambiente Python 3.10 ou superior na raiz e instale
`pip install -e ".[notebooks,notebooks-hires]"`. Instale a variante de PyTorch adequada à sua GPU
conforme a documentação do PyTorch, quando necessário. Inicie o JupyterLab a
partir da raiz (`jupyter lab`) e abra os notebooks; execute as células em ordem.
Os dados originais e os pesos gerados não são incluídos no Git.

Por padrão os dados são lidos de `data/` e os resultados vão para
`outputs/proposta/`. Para um armazenamento remoto ou externo defina, antes de
abrir o Jupyter, `RESISTOR_DATA_ROOT` e `RESISTOR_OUTPUT`. O notebook 2 aceita
`RESISTOR_PROJECT_ROOT` para localizar os manifestos e o código de referência.
Os caminhos esperados dentro da raiz de dados são:

- `sources/roboflow-electronic-components/extracted/{train,valid,test}` para o
  export YOLO do Samu;
- `sources/kaggle-barrettotte/extracted/` para a avaliação complementar de cores.
- `sources/kaggle-eralpozcan/extracted/` e `sources/github-hires/` para o
  teste complementar do segmentador. Suas dependências fazem parte do extra
  `notebooks-hires`.

Na pedrogpu os dados existentes estão em
`/home/pedro/detector-resistores/data`. A execução desta branch foi separada em
`/home/pedro/resistor-proposta-notebooks/outputs`. `RESISTOR_INITIAL_WEIGHTS`
pode apontar para os pesos genéricos locais `yolo11n.pt`. É possível configurar
`RESISTOR_EPOCHS` e `RESISTOR_BATCH`; os padrões são 30 e 8.
Para repetir apenas a avaliação do notebook 1, `RESISTOR_EVALUATE_WEIGHTS`
pode apontar para um checkpoint já treinado na mesma divisão. Não use pesos
treinados com imagens de validação ou teste para afirmar desempenho independente.

Para abrir o Jupyter executando na pedrogpu, crie no computador local o túnel
`ssh -L 8888:127.0.0.1:8888 pedro@192.168.1.57`. Na sessão remota, entre na
cópia desta branch e inicie:

```bash
export RESISTOR_PROJECT_ROOT="$PWD"
export RESISTOR_DATA_ROOT=/home/pedro/detector-resistores/data
export RESISTOR_OUTPUT=/home/pedro/resistor-proposta-notebooks/outputs
/home/pedro/detector-resistores/.venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Abra no navegador local o endereço indicado pelo Jupyter, usando o token
apresentado na sessão remota. O processamento e os dados permanecem na pedrogpu.
Para a demonstração integrada do notebook 2 defina `RESISTOR_IMAGE` com o
caminho de uma imagem após concluir o notebook 1. O notebook 4 aceita a mesma
variável e pode receber um peso de detector em `RESISTOR_YOLO_WEIGHTS`.
Para experimentar o segmentador treinado no notebook 7, aponte
`RESISTOR_HIRES_WEIGHTS` para `segmentacao-faixas-best.pt` na pasta de saída e
use `RESISTOR_SEG_SIZE=256` (ou o tamanho usado no treinamento).
Na pedrogpu, a versão exploratória de 40 épocas está em
`/home/pedro/resistor-proposta-notebooks/outputs/segmentacao-40ep/segmentacao-faixas-best.pt`.
Para repetir apenas a avaliação do notebook 7, use
`RESISTOR_SEG_EVALUATE_WEIGHTS` com o caminho do checkpoint treinado nessa mesma
divisão, e mantenha `RESISTOR_SEG_SIZE` igual ao usado no treinamento.
Para enviar previsões ao notebook 6, configure também `RESISTOR_IMAGE_ID` com
o mesmo id da anotação e `RESISTOR_OWN_PREDICTIONS` com o caminho JSONL de
saída. O notebook grava somente os metadados da previsão, não a imagem.

Para reproduzir a demonstração preliminar de quatro resistores do pacote
MathWorks na pedrogpu, use no notebook 4 `RESISTOR_YOLO_WEIGHTS` apontando para
`/home/pedro/detector-resistores/data/results/general-yolo-5060/train/weights/best.pt`,
`RESISTOR_IMGSZ=960`, `RESISTOR_CONFIDENCE=0.25`, `RESISTOR_IMAGE` para o arquivo `res13clc.jpg` do pacote
MathWorks e `RESISTOR_REFERENCE` para
`data/manifests/mathworks-res13clc-referencia.json`. Essa referência foi anotada
por inspeção visual e contém apenas uma imagem: não é um teste estatístico.

## Protocolo e limites

- O export Samu possui 9.850 imagens. O manifesto anterior contém 2.794 imagens
  com resistores e 3.947 caixas; não confundir imagens com objetos. O export
  não informa valor ou tolerância.
- O notebook 1 agrupa nomes anteriores ao sufixo Roboflow e cópias idênticas
  antes da divisão aproximada 70/15/15. Isso não prova independência de cena ou
  de componentes físicos: o export não fornece essas identidades. É necessária
  revisão manual para uma avaliação independente dessas condições.
- O Barrettotte complementa a avaliação de valores. Seu manifesto antigo usa
  a imagem inteira como caixa; não há caixa de corpo validada. Não interprete
  essa avaliação como localização correta ou como desempenho final de ponta a
  ponta. Grupos por pasta reduzem, mas não eliminam, possíveis dependências.
- O código mede precisão, recall e mAP de localização, contagem exata, erro de
  contagem, tempo e leitura. As leituras recusadas entram no denominador; a
  confiança do detector não é a confiança do valor. A meta de 95% é uma meta,
  não um resultado já obtido.
- As anotações próprias de caixa, cor, orientação e valor por resistor ainda
  são necessárias para validar a saída integrada em imagens com múltiplas
  peças e objetos separados.
- O checkpoint HiRes é de terceiros; a independência do seu treinamento em
  relação ao Eralp não foi estabelecida, portanto esse teste é exploratório.

## Referências usadas

- Caoile, *Automatic Resistor Value Detection Using Computer Vision and Machine
  Learning*, 2026: [artigo](https://zenodo.org/records/20699559). A proposta
  segue a ideia de localizar a região e classificar as faixas; não reivindica
  reprodução integral dos resultados do artigo.
- [Roboflow Samu](https://universe.roboflow.com/samu-drioq/electronic-components-d6uul),
  [Barrettotte](https://www.kaggle.com/datasets/barrettotte/resistors) e
  [Eralp Ozcan](https://www.kaggle.com/datasets/eralpozcan/resistor-dataset).
- [HiRes](https://github.com/HiRes491/HiRes),
  [detector MATLAB](https://www.mathworks.com/matlabcentral/fileexchange/73099-resistor-color-code-value-detector-from-image)
  e [Automatic Component Analyzer](https://github.com/bytesizedengineering/automatic-component-analyzer).
