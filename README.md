# Dashboard — Determinantes do Peso ao Nascer no Brasil (SINASC 2026)

Projeto de Modelagem Estatística: análise descritiva, testes de hipóteses e regressão linear
sobre os microdados do Sistema de Informações sobre Nascidos Vivos, com dashboard interativo.

## Conjunto de Dados

- **Dataset utilizado:** SINASC 2026 — Sistema de Informações sobre Nascidos Vivos (Ministério da
  Saúde / DATASUS)
- **Fonte:** https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINASC/csv/SINASC_2026_csv.zip
- **Formato:** CSV, separador `;`, codificação latin-1, 61 colunas
- **Volume:** 1.504.579 registros (~530 MB), nascimentos de janeiro a agosto de 2026
- **Justificativa da escolha:** Évum censo administrativo de cobertura quase universal, com
  desfecho contínuo adequado a regressão linear (`PESO`, em gramas), variável acionável por
  política pública (número e precocidade das consultas de pré-natal) e relevância sanitária
  direta — o baixo peso ao nascer é o preditor isolado mais forte de mortalidade neonatal e
  infantil, além de indicador pactuado de vigilância em saúde.

## Pergunta de Pesquisa

Quais fatores maternos, gestacionais e assistenciais determinam o peso ao nascer, qual a magnitude
de cada efeito, e quanto se ganharia ao levar a assistência pré-natal ao padrão preconizado pelo
Ministério da Saúde?

## Principais Resultados

| Indicador | Valor |
|---|---|
| Peso médio ao nascer | 3.163,1 g — IC 95% (3.162,2; 3.164,0) |
| Baixo peso ao nascer (< 2.500 g) | 9,04% — IC 95% (9,00; 9,09) |
| Prematuridade (< 37 semanas) | 12,26% |
| Taxa de cesárea | 62,55% |
| Pré-natal adequado (Kotelchuck 4–5) | 80,98% |
| Modelo final | MQO múltiplo, 21 parâmetros, erros-padrão HC3 |
| R² fora da amostra | 0,368 |
| RMSE fora da amostra | 437 g (20,5% melhor que a média) |
| Risco relativo de BPN sob pré-natal inadequado | 1,741 — IC 95% (1,721; 1,761) |

## Tecnologias Utilizadas

| Tecnologia | Tipo | Descrição |
|---|---|---|
| Python 3.14 | Linguagem | Base de toda a análise |
| pandas / numpy | Manipulação de dados | Leitura, limpeza e engenharia de variáveis |
| scipy.stats | Inferência | Testes t de Welch, ANOVA, Levene, Kruskal-Wallis, qui-quadrado, correlações, intervalos de confiança |
| statsmodels | Modelagem | MQO, erros-padrão robustos HC3, Breusch-Pagan, VIF, Tukey HSD, testes F aninhados |
| scikit-learn | Validação | Divisão treino/teste e métricas de desempenho |
| matplotlib / seaborn | Visualização | Gráficos do notebook |
| pyarrow | Persistência | Exportação dos artefatos em Parquet |
| Streamlit / Plotly | Dashboard | Interface interativa (em desenvolvimento) |

## Estrutura do Projeto

```
Projeto-Dashboard/
├── analise.ipynb        # notebook completo: limpeza, EDA, testes, regressão, previsão
├── dashboard.py         # aplicativo Streamlit (7 abas)
├── requirements.txt
├── README.md
└── dados/               # artefatos gerados pelo notebook (Parquet)
    ├── sinasc_amostra.parquet          # amostra estratificada por UF (149.999 registros)
    ├── agregado_uf.parquet             # indicadores por UF + resíduos do modelo agregado
    ├── agregado_regiao.parquet
    ├── agregado_escolaridade.parquet
    ├── agregado_semana.parquet
    ├── serie_mensal.parquet
    ├── prevalencias.parquet            # proporções com IC de Wilson
    ├── testes_hipoteses.parquet        # 6 hipóteses: estatística, gl, p, IC, tamanho de efeito
    ├── coeficientes_m3.parquet         # coeficientes do modelo final com IC e HC3
    ├── comparacao_modelos.parquet
    └── metricas_modelo.parquet
```

## Dashboard

`dashboard.py` é um aplicativo Streamlit que lê exclusivamente os artefatos Parquet de
`dados/` — não depende do CSV bruto nem da reexecução do notebook.

### Abas

| Aba | Conteúdo | Fonte |
|---|---|---|
| Exploração | Distribuição de variável contínua, comparação entre grupos (média com IC, boxplot, violino) e dispersão bivariada com ajuste linear e correlações | Amostra filtrada |
| Território | Indicadores por UF e região ordenados, cruzamento de dois indicadores, resíduos do modelo agregado | Amostra filtrada e população completa |
| Temporal | Série mensal com dois eixos e peso por semana gestacional com faixa de IC | Amostra filtrada ou população completa |
| Prevalências | Intervalos de Wilson dos indicadores binários e recálculo por categoria | População completa e amostra filtrada |
| Testes de hipóteses | Estatística, graus de liberdade, valor-p, tamanho de efeito e IC das seis hipóteses, mais teste t de Welch executado ao vivo sobre o recorte | População completa e amostra filtrada |
| Regressão | Comparação dos cinco modelos, coeficientes do M3 com IC robusto HC3 e diagnóstico de resíduos | População completa e amostra filtrada |
| Simulador | Previsão pontual, intervalo de previsão, probabilidade de baixo peso, curva de sensibilidade e matriz de cenários de pré-natal | Coeficientes do M3 |

### Controles globais

A barra lateral define o nível de confiança (80%, 90%, 95% ou 99%), aplicado a todos os
intervalos recalculados em tempo de execução, e os filtros de região, UF, escolaridade,
raça/cor, sexo, via de parto, adequação do pré-natal, faixa de semanas gestacionais e faixa
de idade materna.

Os filtros atuam sobre as análises derivadas da amostra estratificada de 149.999 registros.
Os resultados calculados sobre a população completa (1.388.675 registros) — prevalências,
testes de hipóteses, coeficientes e métricas dos modelos — vêm pré-computados do notebook e
não respondem aos filtros, o que está indicado na própria interface.

## Executar Localmente

### Pré-requisitos

```
pip install -r requirements.txt
```

### Como rodar o projeto

1. Clone o repositório

```
git clone https://github.com/JJuliaLJ/Projeto-Dashboard.git
cd Projeto-Dashboard
```

2. Para abrir o dashboard, os artefatos de `dados/` já estão versionados:

```
streamlit run dashboard.py
```

3. Para reexecutar a análise do zero, baixe o dataset e descompacte `SINASC_2026.csv` no
   diretório do projeto ou no diretório pai (o notebook procura nos dois, além da subpasta
   `dados/`), e execute o notebook:

```
jupyter notebook analise.ipynb
```

A execução completa leva de 8 a 12 minutos e regenera os arquivos Parquet em `dados/`.


### Roteiro do notebook

| Seção | Conteúdo |
|---|---|
| 1–3 | Leitura seletiva de 25 colunas, todas como texto |
| 4 | Recodificação dos códigos de "ignorado" (9, 99) e filtros de plausibilidade biológica |
| 5 | Engenharia de variáveis: 12 variáveis derivadas |
| 6 | Análise de casos completos com verificação do mecanismo de ausência |
| 7 | Análise exploratória: descritivas, prevalências com IC de Wilson, cortes bivariados, série mensal |
| 8 | Seis testes de hipóteses com graus de liberdade, intervalos de confiança e tamanhos de efeito |
| 9 | Cinco especificações de regressão linear (M1 a M5) |
| 10 | Diagnóstico dos pressupostos: Breusch-Pagan, VIF, HC3, análise de sensibilidade |
| 11 | Previsão fora da amostra, calibração e intervalos de predição |
| 12 | Simulação contrafactual e regressão no nível agregado por UF |
| 13 | Exportação dos artefatos em Parquet para o dashboard |

## Equipe

- Júlia Labad Jatene
- Pedro Andrade Gonçalves de Souza
