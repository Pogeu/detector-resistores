# Benchmark de detecção e leitura de resistores

Este benchmark compara métodos diferentes com o mesmo conjunto de amostras e o
mesmo critério de avaliação. Ele separa localização, faixas, valor nominal,
tolerância, estabilidade em vídeo e velocidade.

## Estratégias

| Estratégia | Tipo | Implementação | Objetivo |
| --- | --- | --- | --- |
| `classical-4` | contorno + HSV + amostragem fixa | disponível | baseline ponta a ponta para 4 faixas |
| `classical-5` | contorno + HSV + amostragem fixa | disponível | baseline ponta a ponta para 5 faixas |
| `crop-classical-4` | recorte conhecido + HSV + amostragem fixa | disponível | baseline de leitura em crops de 4 faixas |
| `crop-classical-5` | recorte conhecido + HSV + amostragem fixa | disponível | baseline de leitura em crops de 5 faixas |
| `classical-lab-pca` | máscara de cor + Lab + PCA | adaptador externo | medir ganho de normalização e orientação |
| `cnn-crop` | CNN em recorte | adaptador externo | testar classificação direta |
| `detector-only` | detector de objetos | adaptador externo | medir apenas localização/contagem |
| `detector-cnn` | detector + CNN | adaptador externo | localização seguida de valor |
| `segmentation-rule` | segmentação + decodificador IEC | adaptador externo | abordagem interpretável |
| `hybrid-4` | detector + segmentação + regras | adaptador externo | abordagem recomendada para 4 faixas |
| `hybrid-5` | detector + segmentação + regras | adaptador externo | abordagem recomendada para 5 faixas |
| `hybrid-auto` | hipótese 4/5/6 + validação | adaptador externo | escolher automaticamente a leitura |
| `hybrid-video` | híbrido + tracking + fusão temporal | adaptador externo | estabilidade em vídeo |

Os adaptadores externos devem gerar o mesmo JSONL de previsões. Assim, YOLO,
U-Net/UNet++, PyTorch, TensorFlow ou um método clássico podem ser comparados sem
mudar as métricas.

## Formato do manifesto

Cada linha representa uma imagem ou um frame de vídeo:

```json
{"id":"img-001","path":"images/img-001.jpg","media_type":"image","objects":[{"bbox":[10,20,120,35],"bands":["brown","black","red","gold"],"ohms":1000,"tolerance_percent":5}]}
```

Para vídeo, use `media_type: "video"`, `source_id` e `frame`. As caixas usam
`[x, y, width, height]`. Os nomes de cor devem ser normalizados para:
`black,brown,red,orange,yellow,green,blue,violet,gray,white,gold,silver`.

O arquivo [manifest.example.jsonl](manifest.example.jsonl) contém exemplos de
imagem e vídeo.

Depois de baixar os datasets, gere os manifestos reais com:

```powershell
python tools/prepare_manifests.py
```

Isso cria manifestos em `data/manifests/` para Eralp (valor por crop),
Barrettotte (valor e faixas por crop) e Alptugaydin (caixas derivadas das
anotações de segmentação). Os manifestos de crop medem leitura de valor; eles
não devem ser usados como benchmark de localização em imagem inteira.

## Execução

No diretório raiz do projeto:

```powershell
python -m resistor_detector.benchmark run `
  --manifest benchmarks/manifest.example.jsonl `
  --strategy classical-4 `
  --root . `
  --output artifacts/predictions-classical-4.jsonl

python -m resistor_detector.benchmark evaluate `
  --ground-truth benchmarks/manifest.example.jsonl `
  --predictions artifacts/predictions-classical-4.jsonl `
  --strategy classical-4 `
  --output artifacts/report-classical-4.json
```

O manifesto de exemplo é apenas um contrato de formato; os arquivos de mídia
precisam existir para executar o baseline.

## Métricas

- `objects`: precision, recall, F1, contagem e MAE de quantidade;
- `bands`: acerto por posição e sequência completa;
- `value`: decodificação, valor nominal, tolerância, objeto correto e amostra completamente correta;
- `video`: consistência temporal e taxa de troca de valor por track;
- `performance`: latência média, P95 e FPS estimado.

Um objeto só é localizado corretamente quando IoU >= 0,50. O relatório também
deve ser executado com IoU 0,75 para evitar que caixas frouxas pareçam boas.

O status de download das fontes está em
[SOURCES_STATUS.md](SOURCES_STATUS.md).

## Matriz de cenários

O arquivo [matrix.json](matrix.json) define os cenários que devem ser cobertos:

- 4, 5 e 6 faixas;
- fundo claro, escuro, PCB e desordenado;
- iluminação uniforme, sombra, reflexo e exposição alterada;
- orientação horizontal, vertical, inclinada e perspectiva;
- foco bom, desfoque, baixa resolução e compressão;
- resistor isolado, múltiplos resistores e oclusão parcial;
- imagem estática e vídeo com movimento.

O teste final deve separar treino, validação e teste por resistor físico e por
sessão de captura. Frames do mesmo vídeo não podem cair em partições diferentes.
