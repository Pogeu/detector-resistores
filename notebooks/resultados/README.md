# Resumos dos experimentos

Arquivos pequenos copiados das execuções na pedrogpu; dados, imagens e pesos
permanecem na máquina remota. A interpretação e as limitações estão em
[../RESULTADOS.md](../RESULTADOS.md).

- `deteccao_resumo.csv` e `map_yolo.json`: avaliação do notebook 1;
- `limiares_validacao.csv` e `teste_limiar_calibrado.json`: notebook 5;
- `preprocessamento_resumo.csv`: comparação clássica do notebook 3;
- `leitura_resumo.csv`: referências e perfis na imagem inteira, notebook 2;
- `segmentacao_15_epocas.json` e `segmentacao_40_epocas.json`: notebook 7,
  com pesos selecionados pela validação e teste sobre recortes anotados;
- `multiplos_segmentador_*.csv`: imagem MathWorks `res13clc.jpg`, usando o
  detector geral antigo e o segmentador indicado no nome. Os três experimentos
  são amostras de uma única imagem, não avaliações estatísticas.
