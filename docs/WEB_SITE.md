# Interface web

O projeto possui uma interface web publicada como **Resistor Lens**:

<https://resistor-lens.magichippo3.chatgpt.site>

O Site está em modo privado por padrão; é necessário entrar com a conta ChatGPT autorizada. A audiência pode ser ampliada depois, se necessário.

Ela permite:

- abrir a câmera traseira do celular;
- capturar uma foto ou enviar uma imagem da galeria;
- detectar e contar os resistores visíveis;
- desenhar caixas sobre os componentes;
- estimar o valor pelas faixas de cores.

## Execução local

O código-fonte do Site fica no checkout privado `site/` e usa apenas HTML, CSS e JavaScript, sem build obrigatório. Para uma prévia local:

```powershell
python -m http.server 4173 --directory site/dist
```

Abra `http://127.0.0.1:4173`. A câmera exige HTTPS ou `localhost`; o upload funciona também em arquivos locais.

## Limitações atuais

A versão publicada executa a análise localmente no navegador. Em fotos de fita com resistores homogêneos, usa consenso entre as leituras para preservar a contagem quando uma faixa fica encoberta por reflexo ou desfoque; nesses itens a confiança aparece como média. Para produção, o próximo passo é conectar a mesma interface ao detector Python/CNN do repositório por uma API HTTPS.
