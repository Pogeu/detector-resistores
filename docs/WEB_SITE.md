# Site Resistor Lens

Aplicação web estática para contar resistores em fotos e estimar o valor de cada componente pelas faixas de cores.

## Acesso

- Site publicado, sem login do ChatGPT: <https://resistor-lens.magichippo3.chatgpt.site>
- O processamento da imagem acontece no navegador; a aplicação não envia automaticamente a foto para um servidor.

## Uso

1. Abra o site no celular ou computador.
2. Toque em **Abrir câmera**, **Enviar imagem** ou arraste uma imagem para a área tracejada.
3. O arquivo é analisado automaticamente. Também existe o botão **Testar imagem exemplo**.
4. Confira a contagem, as caixas numeradas e a lista de valores individuais.

O acesso à câmera depende de HTTPS/localhost e da permissão do navegador. A imagem deve estar em JPG, PNG ou WEBP, com até 20 MB.

## Implementação

O site é buildless e fica em `site/dist/`. O detector usa JavaScript local para:

- segmentar corpos bege por projeção horizontal/vertical, evitando duplicatas;
- identificar faixas por HSV/RGB e suavização do perfil central do corpo;
- decidir a orientação usando a faixa de tolerância dourada/prateada;
- decodificar códigos de 4 ou 5 faixas e exibir baixa confiança quando não há evidência suficiente.

O arquivo de demonstração `site/dist/assets/Electronic-Axial-Lead-Resistors-Array.png` contém seis resistores. O teste local atual encontrou exatamente seis componentes e leu: 47 kΩ, 2,2 kΩ, 200 Ω, 1,5 kΩ, 47 Ω e 29 kΩ.

## Executar localmente

Na raiz do repositório:

```powershell
python -m http.server 4173 --directory site/dist
```

Depois, abra <http://127.0.0.1:4173>. Um servidor HTTP é necessário para que `fetch`, upload e câmera funcionem de forma consistente.

## Limitações conhecidas

Reflexos, baixa resolução, faixas parcialmente ocultas, componentes sobrepostos e resistores muito pequenos podem reduzir a confiança ou exigir uma nova foto. A leitura é uma estimativa visual: confirme valores críticos com multímetro.
