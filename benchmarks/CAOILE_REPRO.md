# Reprodução do baseline de Caoile

## Dataset

Foi usado `data/sources/kaggle-barrettotte/`, a correspondência mais provável do dataset do artigo:

- 1.270 imagens JPG;
- resistores de 4 e 5 bandas;
- 89 valores distintos agrupados a partir dos diretórios de tolerância/potência;
- imagens originais em 1280x720.

O registro Zenodo associado ao artigo contém apenas o PDF. O artigo também não informa filtros da CNN, magnitude das augmentations, seed, taxa de aprendizado ou regra detalhada de split.

## Protocolo implementado

- resize para 64x64 e valores em [0, 1];
- rotação aleatória de 15 graus, flip horizontal e brightness de 0,2;
- duas convoluções 3x3 com ReLU e pooling 2x2;
- 32 e 64 filtros, hipótese necessária por ausência no artigo;
- camada densa de 128 e saída de 89 logits;
- CrossEntropyLoss, equivalente à entropia cruzada categórica;
- Adam com taxa 0,001;
- divisão estratificada 80:20, seed 42;
- batch 32, até 10 épocas e early stopping.

## Execuções

| Variante | Melhor validação | Épocas | Observação |
| --- | ---: | ---: | --- |
| Protocolo aproximado do artigo | 1,57% | 4 | parada após três épocas sem melhora |
| Sem rotação/brightness | 2,36% | 10 | diagnóstico |
| Recorte central 720x720 + augmentations | 1,97% | 10 | aproxima a região-guia das figuras |
| Recorte central + taxa Adam 0,01 | 1,57% | 10 | diagnóstico |

O teste de sanidade conseguiu memorizar um lote com 100% de acurácia; portanto, a implementação e o fluxo de treinamento funcionam. A diferença para os 90,1% reportados no artigo não foi reproduzida com os dados públicos disponíveis. Isso sugere que o dataset experimental original não foi publicado, ou que existem detalhes não documentados no artigo (recorte, split, filtros, seleção das imagens ou vazamento entre treino e validação).

## Comandos

```powershell
$env:PYTHONPATH = "src"
python -m resistor_detector.caoile_cnn train `
  --dataset data/sources/kaggle-barrettotte `
  --output data/results/caoile-cnn.pt `
  --metrics data/results/caoile-cnn-metrics.json
```

Os checkpoints e métricas ficam em `data/results/`, ignorados pelo Git.
