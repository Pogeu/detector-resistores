# Site Resistor Lens

Aplicação web estática para contar resistores em fotos e estimar o valor de cada componente pelas faixas de cores.

## Acesso

- Site publicado: <https://resistor-lens.pogeu.com>
- A imagem é enviada somente durante a análise para a API GPU do projeto, mantida em memória e não persistida em disco.

## Uso

1. Abra o site no celular ou computador.
2. Arraste uma imagem, clique na área de upload ou use **Usar câmera**.
3. Clique em **Analisar agora** para obter a contagem, as caixas numeradas e os valores individuais.
4. Use **Carregar exemplo** para validar a instalação sem precisar enviar uma imagem própria.
5. Abra **Calculadora** para escolher 4 ou 5 faixas e converter valores em cores ou cores em valores.

O acesso à câmera depende de HTTPS/localhost e da permissão do navegador. A imagem deve estar em JPG, PNG ou WEBP, com até 20 MB.

## Implementação

O site é buildless e fica em `site/dist/`. A interface usa JavaScript local para:

- segmentar corpos bege por projeção horizontal/vertical, evitando duplicatas;
- identificar faixas por HSV/RGB e suavização do perfil central do corpo;
- decidir a orientação usando a faixa de tolerância dourada/prateada;
- decodificar códigos de 4 ou 5 faixas e exibir baixa confiança quando não há evidência suficiente.

A inferência principal é executada pela API FastAPI em `server/web_api.py`, usando a GPU do servidor remoto. Os arquivos `server/resistor-api.service`, `server/pogeu-nginx.conf`, `server/pogeu-Dockerfile` e `server/pogeu-cloudflared.yml` documentam a execução do serviço e a publicação em `pogeu.com`.

O arquivo de demonstração `site/dist/assets/Electronic-Axial-Lead-Resistors-Array.png` contém seis resistores. A leitura de referência encontra: 47 kΩ, 2,2 kΩ, 200 Ω, 1,5 kΩ, 47 Ω e 39 kΩ.

## Executar localmente

Na raiz do repositório:

```powershell
python -m http.server 4173 --directory site/dist
```

Depois, abra <http://127.0.0.1:4173>. Um servidor HTTP é necessário para que `fetch`, upload e câmera funcionem de forma consistente.

## Limitações conhecidas

Reflexos, baixa resolução, faixas parcialmente ocultas, componentes sobrepostos e resistores muito pequenos podem reduzir a confiança ou exigir uma nova foto. A leitura é uma estimativa visual: confirme valores críticos com multímetro.
