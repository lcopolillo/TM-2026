# Análise de Sentimento de Críticas de Filmes do IMDB

**Estudo Comparativo de Abordagens Baseadas em Léxico, Aprendizagem Automática e Modelos de Língua**

TMCD 2025/2026 — 2.º Semestre
Departamento de Ciências e Tecnologias da Informação
ISCTE — Instituto Universitário de Lisboa

**Luiza Coelho · Pedro Louro · Tiago Vieira** — Abril 2026

---

## Resumo

Este trabalho compara várias abordagens de análise de sentimento binária (positivo/negativo) aplicadas ao conjunto de dados IMDB Movie Reviews. O conjunto de treino tem 41 750 críticas e o de teste 2 000, com distribuição equilibrada entre as duas classes.

Foram implementadas e avaliadas quatro famílias de métodos: ferramentas baseadas em léxicos e regras (TextBlob, VADER e Stanza), com acurácias entre 70% e 83%; um transformer pré-treinado sem ajuste fino (DistilBERT SST-2), com 90%; um classificador baseado no léxico NRC EmoLex com e sem tratamento da negação, entre 64% e 65%; modelos clássicos de aprendizagem automática (Regressão Logística, Naive Bayes, SVM) com representações BoW e TF-IDF, chegando a 90,25%, e o DistilBERT com ajuste fino, com 94,75%; e o modelo Claude Haiku com instruções em linguagem natural, que alcançou 96,2% com uma instrução few-shot numa amostra de 500 críticas.

Os resultados mostram que modelos de língua de grande escala, sem qualquer treino específico nos dados, conseguem igualar ou superar o ajuste fino de transformers.

**Contribuições:** Luiza Coelho: 33,3% — tarefas 2.1.1 (Stanza) e 2.2 (NRC). Pedro Louro: 33,3% — tarefas 2.1.2 (DistilBERT baseline) e 2.3 (modelos clássicos e ajuste fino). Tiago Vieira: 33,3% — tarefa 2.4 (LLM prompting), coordenação e utilitários partilhados.

---

## Índice

- [1. Introdução](#1-introdução)
- [2. Trabalho Relacionado](#2-trabalho-relacionado)
- [3. Dados](#3-dados)
- [4. Metodologia](#4-metodologia)
  - [4.1 Ferramentas Existentes de Análise de Sentimento](#41-ferramentas-existentes-de-análise-de-sentimento)
  - [4.2 Classificador com o Léxico NRC EmoLex](#42-classificador-com-o-léxico-nrc-emolex)
  - [4.3 Aprendizagem Automática Clássica](#43-aprendizagem-automática-clássica)
  - [4.4 Ajuste Fino do DistilBERT](#44-ajuste-fino-do-distilbert)
  - [4.5 Prompting com Modelos de Língua de Grande Escala](#45-prompting-com-modelos-de-língua-de-grande-escala)
- [5. Resultados e Discussão](#5-resultados-e-discussão)
- [6. Conclusões](#6-conclusões)
- [Apêndice A — Definição das Instruções](#apêndice-a--definição-das-instruções)

---

## Lista de Figuras

- Figura 1 — Distribuição das classes no treino e no teste
- Figura 2 — Distribuição do número de tokens por classe (treino)
- Figura 3 — 20 palavras mais frequentes por classe (treino, sem stopwords)
- Figura 4 — Nuvens de palavras — críticas positivas e negativas
- Figura 5 — Acurácia e F1 das ferramentas de referência
- Figura 6 — Distribuição de confiança do DistilBERT pré-treinado (SST-2)
- Figura 7 — Resultados do classificador NRC com e sem negação
- Figura 8 — Acurácia dos modelos clássicos por representação e classificador
- Figura 9 — Perda de treino e acurácia de validação do DistilBERT por época
- Figura 10 — Métricas do Claude Haiku por tipo de instrução
- Figura 11 — Todas as abordagens ordenadas por acurácia

---

## 1. Introdução

A análise de sentimento consiste em identificar a polaridade (positiva ou negativa) expressa num texto. É uma das tarefas mais estudadas em Processamento de Linguagem Natural, com aplicações em redes sociais, plataformas de avaliações e notícias.

As críticas de filmes do IMDB são um caso de teste especialmente interessante porque os textos são longos (em média 175 palavras), variados e contêm construções complexas como ironia, negação e vocabulário específico do domínio do cinema. Estas características tornam a tarefa mais difícil e mais realista do que classificar textos curtos como tweets.

Neste trabalho implementámos e comparámos quatro abordagens diferentes: ferramentas de análise de sentimento já existentes (TextBlob, VADER, Stanza e DistilBERT pré-treinado), um classificador baseado no léxico NRC EmoLex, modelos clássicos de aprendizagem automática e o ajuste fino do DistilBERT, e por fim o modelo Claude Haiku com diferentes tipos de instruções. Todos os resultados foram guardados em `results/all_results.csv` e os utilitários partilhados estão em `src/utils.py`.

---

## 2. Trabalho Relacionado

*(A completar — incluir ≥ 6 artigos sobre: léxicos de sentimento, VADER/TextBlob, dataset IMDB, DistilBERT/BERT, aprendizagem automática clássica para sentimento e prompting few-shot com LLMs.)*

---

## 3. Dados

O conjunto de dados IMDB Movie Reviews foi fornecido em dois ficheiros CSV — treino e teste — com uma coluna de texto e uma coluna de etiqueta (`pos` ou `neg`).

| Split | Total  | Pos    | Neg    | Média de palavras |
|-------|--------|--------|--------|-------------------|
| Treino | 41 750 | 20 700 | 21 050 | 178               |
| Teste  |  2 000 |  1 022 |    978 | 175               |

Os dois conjuntos estão quase perfeitamente equilibrados (≈50/50), o que torna a acurácia uma métrica válida como métrica principal e coloca a linha de base de maioria em 51,1%.

![Figura 1 — Distribuição das classes no treino e no teste](../results/fig_class_distribution.png)

As críticas são muito mais longas do que outros conjuntos de sentimento como tweets (≈20 palavras). Isso afeta cada método de forma diferente: as ferramentas baseadas em léxico acumulam muito ruído ao longo de 175 palavras; os transformers truncam críticas com mais de 512 tokens (cerca de 15% do conjunto); os modelos clássicos beneficiam do vocabulário rico por amostra; e o Claude Haiku recebeu as críticas truncadas a 1 500 caracteres por razões de custo.

![Figura 2 — Distribuição do número de tokens por classe](../results/fig_token_distribution.png)

![Figura 3 — 20 palavras mais frequentes por classe](../results/fig_top_words.png)

![Figura 4 — Nuvens de palavras — positivas (esquerda) e negativas (direita)](../results/fig_wordclouds.png)

---

## 4. Metodologia

### 4.1 Ferramentas Existentes de Análise de Sentimento

Como ponto de partida, aplicámos quatro ferramentas ao conjunto de teste sem qualquer treino adicional, para estabelecer uma linha de base.

O **TextBlob** calcula a polaridade média das palavras de cada texto com base num léxico interno. Valores acima de zero são classificados como positivos. Obteve 70,0% de acurácia, com cobertura alta (0,944) mas precisão baixa (0,640) — tende a classificar demasiadas críticas como positivas porque o seu léxico foi construído a partir de texto geral da internet.

O **VADER** foi criado para textos curtos de redes sociais e usa regras adicionais para capitalização, pontuação e advérbios. Obteve 70,2% de acurácia, resultado idêntico ao TextBlob. As suas regras perdem eficácia em textos longos como as críticas do IMDB.

O **Stanza** analisa o sentimento frase a frase com um modelo neural treinado no Stanford Sentiment Treebank. A média das pontuações por frase é usada para classificar cada crítica. Obteve 83,4% de acurácia, uma melhoria de 13 pontos face ao VADER. Ao considerar o contexto dentro de cada frase em vez de palavras individuais, consegue resultados substancialmente melhores.

O **DistilBERT pré-treinado (SST-2)** foi avaliado com o modelo `distilbert-base-uncased-finetuned-sst-2-english` do Hugging Face, sem treino adicional com dados do IMDB. As críticas foram truncadas a 512 tokens. Obteve 90,0% de acurácia, o melhor resultado entre as ferramentas existentes, mostrando que um transformer pré-treinado generaliza bem mesmo para um domínio diferente.

| Abordagem                       | Acurácia | Precisão | Cobertura | F1     |
|---------------------------------|----------|----------|-----------|--------|
| DistilBERT (pré-treinado, SST-2)| 0,9000   | 0,9228   | 0,8777    | 0,8997 |
| Stanza                          | 0,8335   | 0,9333   | 0,7260    | 0,8167 |
| VADER                           | 0,7015   | 0,6604   | 0,8562    | 0,7456 |
| TextBlob                        | 0,7000   | 0,6399   | 0,9442    | 0,7628 |

![Figura 5 — Acurácia e F1 das ferramentas de referência](../results/fig_lexicon_rules.png)

![Figura 6 — Distribuição de confiança do DistilBERT SST-2](../results/fig_distilbert_baseline_confidence.png)

---

### 4.2 Classificador com o Léxico NRC EmoLex

O NRC EmoLex associa 14 182 palavras inglesas a indicadores de polaridade positiva e negativa. Para cada crítica, contámos quantas palavras são positivas (P) e quantas são negativas (N). Se P > N, a crítica é positiva; caso contrário, é negativa; em empate, atribuímos negativo.

Antes da classificação, as críticas foram convertidas para minúsculas e lematizadas com o `WordNetLemmatizer` do NLTK, para que formas flexionadas como "loved" sejam reconhecidas como "love" e encontradas no léxico.

Para o tratamento da negação, após palavras como "not", "never" ou "n't", os três tokens seguintes têm a polaridade invertida — por exemplo, "good" após "not" passa a contar como negativo. Este passo é feito antes de remover stopwords, para que as palavras de negação não sejam eliminadas primeiro.

Ambas as configurações ficam acima da linha de base de 51,1%, mas são os resultados mais fracos de todas as tarefas. O classificador tende a prever positivo com demasiada frequência (cobertura ≥ 0,81), provavelmente porque até as críticas negativas usam vocabulário positivo ao descrever cenas ou atuações. A negação melhora a acurácia em 0,9 pp, mas o maior problema é a cobertura do léxico: muitas palavras típicas das críticas de filmes — nomes de personagens, expressões informais, termos de género — não existem no NRC e não contribuem para a classificação.

![Figura 7 — Resultados do classificador NRC com e sem negação](../results/fig_nrc_lexicon.png)

---

### 4.3 Aprendizagem Automática Clássica

Os modelos foram treinados com 41 750 críticas e avaliados nas 2 000 do conjunto de teste, usando o `scikit-learn`. Testámos diferentes combinações de pré-processamento, representação de texto e classificadores.

O pré-processamento foi implementado como um pipeline configurável com cinco passos: conversão para minúsculas; remoção de pontuação; marcação de negação (sufixo `_NEG` nos três tokens após palavras como "not" ou "never", aplicada antes de remover stopwords); remoção de stopwords (excluindo as palavras de negação); e lematização. Foram comparadas três configurações: só minúsculas; minúsculas com stopwords e lematização; e o pipeline completo com negação.

Para a representação do texto usámos três abordagens com as 10 000 palavras mais frequentes do treino. O Bag-of-Words (BoW) conta o número de ocorrências de cada palavra. O TF-IDF unigramas pondera as palavras pela sua raridade — palavras muito comuns como "film" recebem menos peso — reduzindo o ruído. O TF-IDF bigramas estende o anterior com pares de palavras consecutivas como "not bad" ou "highly recommended", que têm um significado diferente de cada palavra separada.

Avaliámos três classificadores. A Regressão Logística aprende uma fronteira de separação linear com regularização L2, sendo um dos melhores modelos lineares para classificação de texto. O Naive Bayes Multinomial classifica com base nas probabilidades de cada palavra por classe, assumindo independência entre elas — o que o torna eficiente mas menos preciso em textos longos. O SVM Linear encontra a fronteira que maximiza a margem entre as duas classes, sendo robusto em espaços de alta dimensão. Os hiperparâmetros de todos os modelos foram selecionados por validação cruzada de 5 partições no conjunto de treino.

| Modelo         | Características              | Acurácia | Precisão | Cobertura | F1     |
|----------------|------------------------------|----------|----------|-----------|--------|
| SVM (C=0,1) ★  | TF-IDF bigramas + negação    | 0,9025   | 0,9034   | 0,9061    | 0,9047 |
| SVM (C=1)      | TF-IDF bigramas + negação    | 0,9010   | 0,9112   | 0,8933    | 0,9022 |
| LR (C=1)       | TF-IDF bigramas              | 0,9000   | 0,9014   | 0,9031    | 0,9022 |
| SVM (C=1)      | TF-IDF unigramas             | 0,9000   | 0,9014   | 0,9031    | 0,9022 |
| LR (C=1)       | TF-IDF unigramas             | 0,8925   | 0,8929   | 0,8973    | 0,8951 |
| LR (C=1)       | BoW + stopwords + lema       | 0,8825   | 0,8885   | 0,8806    | 0,8845 |
| NB (α=1)       | BoW + stopwords + lema       | 0,8580   | 0,8758   | 0,8415    | 0,8583 |
| SVM (C=1)      | BoW + stopwords + lema       | 0,8540   | 0,8614   | 0,8513    | 0,8563 |

★ Melhor modelo por validação cruzada (C ∈ {0,1, 0,5, 1, 5, 10}).

![Figura 8 — Acurácia dos modelos clássicos por representação e classificador](../results/fig_classical_ml.png)

Os resultados mostram que a representação do texto tem mais impacto do que o classificador: mudar de BoW para TF-IDF bigramas melhora a acurácia em quase 2 pp mantendo o mesmo classificador. A marcação de negação acrescenta +0,25 pp ao melhor modelo — um ganho modesto mas consistente. O Naive Bayes é o mais fraco dos três, ficando 3 pp abaixo da Regressão Logística com as mesmas características, o que se explica pela sua incapacidade de capturar relações entre palavras.

---

### 4.4 Ajuste Fino do DistilBERT

O DistilBERT pré-treinado no SST-2 foi treinado com frases curtas muito diferentes das críticas longas do IMDB. Fizemos o ajuste fino do modelo `distilbert-base-uncased` com o conjunto de treino completo (41 750 críticas) usando a API `Trainer` do Hugging Face, com truncagem a 512 tokens e padding dinâmico por lote.

O treino foi feito em CPU (o Apple MPS foi desativado por problemas de memória), com 3 épocas, taxa de aprendizagem de 2×10⁻⁵, tamanho de lote de 16 e paragem antecipada se a acurácia de validação não melhorasse numa época. O treino demorou cerca de 6,5 horas, com uma perda final de 0,146.

| Modelo                              | Acurácia | Precisão | Cobertura | F1     |
|-------------------------------------|----------|----------|-----------|--------|
| DistilBERT (ajuste fino, 3 épocas)  | 0,9475   | 0,9508   | 0,9462    | 0,9485 |
| DistilBERT (pré-treinado, SST-2)    | 0,9000   | 0,9228   | 0,8777    | 0,8997 |
| Ganho                               | +0,0475  | +0,0280  | +0,0685   | +0,0488|

O ajuste fino melhorou todos os resultados, com o maior ganho na cobertura (+6,85 pp). Isto mostra que o modelo pré-treinado no SST-2 errava bastante ao não identificar críticas positivas: no SST-2 o sentimento é expresso em frases curtas e diretas, enquanto nas críticas do IMDB o texto mistura positivo e negativo ao longo de vários parágrafos antes de chegar a uma conclusão. O ajuste fino ensinou o modelo a interpretar essa estrutura mais longa.

![Figura 9 — Perda de treino e acurácia de validação por época](../results/fig_finetuning_curves.png)

---

### 4.5 Prompting com Modelos de Língua de Grande Escala

Nesta tarefa usámos o modelo `claude-haiku-4-5-20251001` da Anthropic para classificar o sentimento das críticas apenas com instruções em linguagem natural, sem nenhum treino ou ajuste de parâmetros. Por razões de custo, a experiência foi feita numa amostra de 500 críticas (250 positivas e 250 negativas, com semente 42), em vez das 2 000 das outras tarefas. Cada crítica foi truncada a 1 500 caracteres e o modelo respondeu com uma única palavra (max_tokens=10). Foram feitas 1 500 chamadas sem erros.

Testámos três tipos de instrução. A **instrução genérica** pede simplesmente para classificar o sentimento como positivo ou negativo, sem contexto adicional. A **instrução com contexto de domínio** especifica que o texto é uma crítica de filme e que o modelo deve agir como analista de cinema, ajudando-o a focar no domínio correto. A **instrução few-shot** fornece seis exemplos classificados (três positivos e três negativos, retirados do conjunto de treino com semente 42) antes de cada crítica, para mostrar ao modelo o tipo de linguagem e o formato esperado. Os textos completos das instruções estão no Apêndice A.

| Instrução        | Acurácia   | Precisão | Cobertura  | F1         |
|------------------|------------|----------|------------|------------|
| Few-shot         | **0,9620** | 0,9529   | **0,9720** | **0,9624** |
| Domínio          | 0,9600     | 0,9637   | 0,9560     | 0,9598     |
| Genérica         | 0,9580     | 0,9751   | 0,9400     | 0,9572     |

![Figura 10 — Métricas do Claude Haiku por tipo de instrução](../results/fig_llm_prompting.png)

As três instruções obtiveram acurácias acima de 95,8%, com a few-shot a alcançar 96,2%. As diferenças entre estratégias são pequenas (menos de 0,5 pp), o que sugere que o modelo já tem um bom entendimento do domínio por si mesmo e que o contexto adicional afina mas não transforma as suas previsões. Estes resultados devem ser comparados com cautela com os das outras tarefas, dado o menor tamanho da amostra.

---

## 5. Resultados e Discussão

A Tabela 1 apresenta o melhor resultado de cada abordagem, ordenado por acurácia. Para a tarefa 2.3 incluímos o melhor modelo clássico e o DistilBERT com ajuste fino separadamente, por serem abordagens distintas.

**Tabela 1 — Melhor resultado por abordagem, ordenado por acurácia.**

| Tarefa | Abordagem                            | Acurácia   | Precisão | Cobertura | F1     |
|--------|--------------------------------------|------------|----------|-----------|--------|
| 2.4    | Claude Haiku — few-shot †            | **0,9620** | 0,9529   | 0,9720    | 0,9624 |
| 2.3    | DistilBERT (ajuste fino, 3 épocas)   | 0,9475     | 0,9508   | 0,9462    | 0,9485 |
| 2.3    | SVM — TF-IDF bigramas + negação      | 0,9025     | 0,9034   | 0,9061    | 0,9047 |
| 2.1.2  | DistilBERT (pré-treinado, SST-2)     | 0,9000     | 0,9228   | 0,8777    | 0,8997 |
| 2.1.1  | Stanza                               | 0,8335     | 0,9333   | 0,7260    | 0,8167 |
| 2.1.1  | VADER                                | 0,7015     | 0,6604   | 0,8562    | 0,7456 |
| 2.1.1  | TextBlob                             | 0,7000     | 0,6399   | 0,9442    | 0,7628 |
| 2.2    | Léxico NRC + Negação                 | 0,6550     | 0,6254   | 0,8102    | 0,7059 |

† Avaliado numa amostra de 500 críticas (250 pos + 250 neg).

![Figura 11 — Todas as abordagens ordenadas por acurácia](../results/fig_all_results.png)

Os resultados cobrem uma diferença de 31 pontos percentuais entre a pior e a melhor abordagem, e agrupam-se naturalmente em três níveis.

O grupo mais fraco inclui o classificador NRC (65,5%), o TextBlob e o VADER (≈70%). Estas abordagens atribuem polaridade palavra a palavra, sem ter em conta o contexto. Numa crítica de 175 palavras que mistura observações positivas e negativas antes de chegar a uma conclusão, este método não é suficiente. O Stanza (83,4%) fica acima dos outros neste grupo porque analisa frase a frase em vez de palavra a palavra — uma diferença de 13 pp que mostra o impacto de ter algum contexto na classificação.

O grupo intermédio reúne o DistilBERT pré-treinado (90,0%) com os melhores modelos clássicos (até 90,25%). O facto de um SVM com TF-IDF bigramas e negação atingir o mesmo resultado que um transformer com 66 milhões de parâmetros é um resultado relevante: para um problema binário e equilibrado como este, um modelo clássico bem configurado é igualmente eficaz e muito mais rápido de treinar (menos de 2 minutos em CPU).

O grupo mais forte inclui o DistilBERT com ajuste fino (94,75%) e o Claude Haiku com few-shot (96,2%). O ajuste fino melhora especialmente a cobertura (+6,85 pp), corrigindo a tendência do modelo de não reconhecer críticas positivas quando aplicado fora do seu domínio de treino. O Claude Haiku, sem nenhum treino adicional, obtém o melhor resultado apenas com a instrução certa. A diferença de 1,45 pp entre os dois deve ser interpretada com cuidado, dado que o Claude Haiku foi avaliado em apenas 500 amostras.

---

## 6. Conclusões

Este trabalho comparou abordagens de análise de sentimento de complexidade crescente, desde ferramentas baseadas em listas de palavras até modelos de língua de grande escala, no conjunto de dados IMDB Movie Reviews.

As abordagens mais simples — NRC EmoLex (65%), TextBlob e VADER (≈70%) — mostram os limites de classificar sentimento palavra a palavra em textos longos. A cobertura limitada do léxico NRC e a falta de contexto das ferramentas de regras são os principais fatores que limitam o seu desempenho. O Stanza (83,4%) melhora significativamente por operar ao nível da frase, mas mantém erros em críticas onde o sentimento global não é uniforme.

A aprendizagem automática clássica com TF-IDF bigramas e negação alcança 90,25%, igualando o DistilBERT pré-treinado (90,0%) com muito menos custo computacional. Este é provavelmente o resultado mais prático do trabalho: para um problema binário e equilibrado, um SVM bem configurado é uma alternativa competitiva a modelos muito maiores.

O ajuste fino do DistilBERT (94,75%) demonstra o valor de adaptar um modelo ao domínio específico. O maior ganho foi na cobertura, corrigindo o viés do modelo pré-treinado no SST-2 que subestimava críticas positivas longas.

O Claude Haiku com instrução few-shot alcançou o melhor resultado (96,2%) sem qualquer treino nos dados. As três estratégias de instrução tiveram desempenhos muito próximos (diferença < 0,5 pp), sugerindo que o modelo já conhece bem o domínio. Este resultado deve ser interpretado com cautela dado o tamanho reduzido da amostra (500 críticas).

Como trabalho futuro, seria interessante avaliar o Claude Haiku nas 2 000 críticas do conjunto de teste para uma comparação direta com as outras abordagens. Combinar o léxico NRC com um segundo léxico como o SentiWordNet poderia melhorar a cobertura do classificador lexical. O ajuste fino em GPU permitiria explorar mais configurações de treino e modelos maiores como o RoBERTa.

---

## Apêndice A — Definição das Instruções

As seguintes instruções foram usadas nas experiências com o Claude Haiku (Tarefa 2.4). O texto de cada crítica foi truncado a 1 500 caracteres. Todas as chamadas usaram `max_tokens=10` e temperatura por defeito.

### Instrução 1 — Genérica

```
Classify the sentiment of the following text as "positive" or "negative".
Reply with only one word.

Text: {review}
```

### Instrução 2 — Com contexto de domínio

```
You are analyzing movie reviews. Classify the sentiment of the following
movie review as "positive" or "negative".
Reply with only one word.

Review: {review}
```

### Instrução 3 — Few-shot

```
Classify movie review sentiment as "positive" or "negative".

Examples:
[POS] {example_pos_1}
[NEG] {example_neg_1}
[POS] {example_pos_2}
[NEG] {example_neg_2}
[POS] {example_pos_3}
[NEG] {example_neg_3}

Now classify:
{review}

Reply with only one word: positive or negative.
```

Os seis exemplos foram retirados do conjunto de treino (semente 42): 3 positivos e 3 negativos, truncados a 300 caracteres.
