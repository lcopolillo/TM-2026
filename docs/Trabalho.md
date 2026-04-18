# TMCD — Trabalho
**Dep. de Ciências e Tecnologias da Informação**
Ano letivo 2025/2026 | 2º Semestre | Versão 1.1

---

> O trabalho deverá ser realizado em **grupos de 2, 3** (aconselhado) **ou 4 elementos**.
> O trabalho deve ser entregue até ao dia **24 de abril**. As apresentações serão realizadas nos dias **29 e 30 de abril**, durante a aula.

---

## Análise de Sentimento

A tarefa de Análise de Sentimento a partir de um texto é amplamente conhecida na área de Text Mining. A ideia consiste em analisar o texto e identificar qual o sentimento aí presente. Exemplos:

> [**NEG**] *If you haven't seen this, it's terrible. It is pure trash. I saw this about 17 years ago, and I'm still screwed up from it. If you haven't seen this, it's terrible.*

> [**POS**] *Drum scene is wild! Cook, Jr. is unsung hero of this and many movies. Fantastic actor, great flick. A few twists that keep you moving. A must-see. Drum scene is wild!*

Apesar da classificação de um texto poder ser feita a vários níveis, tais como *positivo*, *negativo*, *neutro*, *muito positivo* ou *muito negativo*, este problema é muitas vezes abordado como uma tarefa de classificação binária, em que se consideram apenas as classes *positivo* e *negativo*.

Este trabalho pretende explorar vários métodos de classificação de sentimentos em textos e aplicá-los a vários tipos de dados, tais como *tweets* e *reviews* de filmes, *reviews* de produtos, etc.

---

## 1 Dados

Para realizar este trabalho estão disponíveis no Moodle vários conjuntos de dados em Inglês, previamente separados em **treino** e **teste**. Cada um dos conjuntos apresenta características e problemas diferentes, o que poderá implicar diferentes tipos de pré-processamento. Cada um dos grupos deverá utilizar um conjunto diferente, por forma a promover a diversidade de trabalhos.

1. **Tweets_EN_sentiment**: Conjunto não balanceado, com cerca de 50.000 *tweets* em Inglês, etiquetados com `pos` e `neg`.
2. **amazon_reviews**: Conjunto não balanceado, com cerca de 50.000 *reviews* de produtos da empresa *Amazon*, anotadas com as etiquetas `positive` e `negative`.
3. **imdb_reviews**: Conjunto não balanceado, com cerca de 44.000 *reviews* de filmes da IMDB, anotadas com as etiquetas `pos` e `neg`.
4. **rotten_tomatoes**: Conjunto com cerca de 6.500 frases de *reviews* de filmes, anotadas com as 3 etiquetas: `positive`, `negative` e `neutral`.

Para obter os dados poderá optar por uma das seguintes soluções:

1. Cada um dos ficheiros individuais está também disponível junto com o enunciado do trabalho na secção do Moodle: *Avaliação ▷ Trabalho - Análise de Sentimento*.
2. Poderá também obter todos os dados disponíveis, seguindo as instruções que estão na secção do Moodle: *Conteúdo ▷ Acesso aos dados dos exemplos práticos*. Ao descomprimir o ficheiro `tm-data.zip`, todos os ficheiros se encontram dentro da diretoria `data/en-sentiment`.

*Observação.* Não deve disponibilizar publicamente nenhum dos conjuntos de dados fornecido.

---

## 2 Tarefas

Deverá começar por fazer uma pesquisa bibliográfica e identificar uma pequena coleção de artigos científicos, com pelo menos 2×*n* artigos, em que *n é o nº de elementos do grupo*. Os artigos devem estar relacionados com o trabalho do grupo de alguma forma, por exemplo: usam uma base de dados semelhante, executam uma tarefa da mesma natureza, ou reportam abordagens relevantes para o trabalho do grupo. Para cada um dos trabalhos, deverá identificar as principais abordagens utilizadas e os principais resultados obtidos, de forma a conseguir compará-los com o que o grupo faz no seu trabalho. Depois de fazer a análise da literatura, execute as seguintes tarefas.

### 2.1 Aplicação de Modelos Existentes

Inicialmente, deverá aplicar alguns modelos existentes para análise de sentimento, de forma a obter um desempenho de referência (*baseline*) para o seu trabalho. Para isso, deverá aplicar os modelos escolhidos aos **dados de teste** e avaliar os resultados usando métricas adequadas. Note que a taxa de acerto (*accuracy*) poderá ser adequada para problemas em que as classes têm igual importância, mas para outros poderá fazer sentido usar outras métricas, tais como: precisão, cobertura e medida F₁. Deverá explorar, pelo menos, as abordagens indicadas nas secções seguintes.

#### 2.1.1 Modelos Baseados em Léxicos e Regras

Os modelos baseados em léxicos e regras utilizam listas de palavras anotadas com polaridade (positiva ou negativa) e aplicam regras para determinar o sentimento global do texto. Pode explorar o uso de ferramentas como o TextBlob, o VADER Sentiment Analysis ou Stanza, entre outras.

Um grupo constituído por *n* elementos deve aplicar e comparar pelo menos *n* ferramentas diferentes. Escolha e aplique as ferramentas aos seus dados de teste e avalie os resultados obtidos utilizando as métricas adotadas. Note que a *taxa de acerto (accuracy)* poderá ser a mais adequada para problemas binários, em que ambas as classes têm igual importância. Para outros problemas poderá fazer sentido usar outras métricas, tais como: *precisão*, *cobertura* e *medida F1*.

#### 2.1.2 Modelos Baseados em Transformadores

Os modelos baseados em transformadores têm demonstrado elevado desempenho em várias tarefas de Processamento Computacional da Língua, incluindo a análise de sentimento. Explore o uso de um destes modelos, pré-treinado para análise de sentimento. Pode usar, por exemplo, o modelo `distilbert-base-uncased-finetuned-sst-2-english` disponível na plataforma Hugging Face.

### 2.2 Aplicação de um Léxico de Sentimentos

Pretende-se agora criar um classificador de sentimentos muito simples, baseado num léxico. Para tal, pode utilizar o NRC Word-Emotion Association Lexicon (EmoLex) que se encontra disponível na diretoria `./data/en/NCR-lexicon.csv`. O ficheiro original é um ficheiro do tipo CSV que tem o seguinte conteúdo, embora apenas as primeiras 3 colunas do ficheiro — *English*, *Positive*, *Negative* — sejam suficientes para este trabalho:

| English     | Positive | Negative | Anger | Anticipation | Disgust | Fear | ... |
|-------------|----------|----------|-------|--------------|---------|------|-----|
| abacus      | 0        | 0        | 0     | 0            | 0       | 0    |     |
| abandoned   | 0        | 1        | 1     | 0            | 0       | 1    |     |
| abandonment | 0        | 1        | 1     | 0            | 0       | 1    |     |
| abba        | 1        | 0        | 0     | 0            | 0       | 0    |     |
| ...         |          |          |       |              |         |      |     |

Mais uma vez, pretende-se fazer a classificação dos **dados de teste**. Deverá procurar alguma bibliografia que reporte a utilização de léxicos de polaridade por forma a decidir qual a melhor abordagem. No entanto, a ideia geral é quantificar quantas palavras do texto a analisar estão classificadas como *Positive* e *Negative* e, consoante a classe mais frequente, decidir qual a classificação final a atribuir ao texto.

Poderá utilizar outros léxicos alternativos e deverá também fazer algum pré-processamento, de forma a melhorar o desempenho obtido. Deve fazer pelo menos duas experiências diferentes: **sem** e **com** tratamento da **negação**, comentando os respetivos resultados obtidos.

*Observação.* No caso do grupo ter mais do que 3 elementos, deverá fazer todas as experiências com dois léxicos diferentes.

### 2.3 Treino de Modelos de Aprendizagem Automática

Deverá agora tentar treinar um ou mais modelos, com o objetivo de tentar melhorar os resultados de classificação obtidos:

1. Treine um modelo com os dados de treino, usando uma abordagem de classificação automática à sua escolha, tal como por exemplo *scikit-learn*, etc.
2. Aplique o modelo que construiu ao conjunto de teste, avalie os resultados obtidos e compare-os com os obtidos na tarefa anterior.

Para esta tarefa, deverá fazer múltiplas experiências, optando por usar diferentes tipos de abordagens:

1. Abordagens clássicas de aprendizagem automática (e.g. regressão logística, *naive Bayes*, árvores de decisão, máquinas de vetores de suporte, redes neuronais, etc.), usando como *features* representações dos documentos baseadas em *bag-of-words*, *TF-IDF* ou *embeddings*.
2. Ajuste (*fine-tuning*) de modelos pré-treinados baseados em transformadores (e.g. BERT, RoBERTa, DistilBERT, etc.).

Note que, no primeiro caso, o pré-processamento dos textos e a seleção das *features* a usar são etapas importantes, pelo que deverá explorar diferentes técnicas e documentar as suas escolhas. Por outro lado, no segundo caso, os recursos computacionais e temporais necessários para treinar estes modelos podem ser significativos, pelo que deverá planear bem a sua abordagem.

*Observação.* Pode basear o seu trabalho em receitas disponíveis na Internet, no entanto, deve fazer referência a todo o tipo de material que utilizar, incluindo o endereço onde se encontra a respetiva informação.

### 2.4 Utilização de Modelos Baseados em Instruções

Por fim, deverá explorar a utilização de modelos de língua baseados em instruções (*instruction-based models*), tais como GPT, Gemini, Claude ou DeepSeek, para realizar a tarefa de análise de sentimento.

Para além do modelo, a instrução (*prompt*) utilizada é um aspeto fundamental para o sucesso deste tipo de abordagens. Deverá explorar diferentes tipos de instruções, avaliando os resultados obtidos com cada uma delas. Nomeadamente, deverá explorar:

1. Uma instrução genérica para análise de sentimento.
2. Incluir uma descrição do domínio de aplicação.
3. Incluir exemplos classificados.

**Nota:** A utilização deste tipo de modelos através de APIs costuma ter custos associados. As plataformas OpenRouter e iaedu.pt disponibilizam acesso gratuito a alguns modelos, com limitações. Alternativamente, poderá usar as versões interativas disponíveis gratuitamente, enviando os textos de teste para análise. No entanto, isso implica algum trabalho adicional na submissão dos textos e na recolha dos resultados.

---

## Submissões, relatório e avaliação

O trabalho deverá ser entregue através do Moodle. Para tal, todos os elementos do deverão inscrever-se num dos grupos disponíveis para realização do Trabalho. O grupo deverá enviar um ficheiro `zip` contendo o relatório e o código desenvolvido. A apresentação poderá ser entregue mais tarde, até às 17h do dia da apresentação (entregas depois desta hora serão penalizadas).

### Relatório

O grupo deverá produzir um relatório a descrever o trabalho realizado, com um **máximo de 10+*n* páginas (+ anexos)**, em que *n* é o *nº de elementos do grupo*, em formato `pdf`, usando o template Springer LNCS (Microsoft Word ou LaTeX). O relatório deverá obrigatoriamente seguir a seguinte estrutura:

- Título; Autores; Resumo
- 1) Introdução
- 2) Trabalho relacionado
- 3) Dados utilizados (descrição e caracterização)
- 4) Trabalho realizado, descrevendo os procedimentos efetuados, as opções tomadas e os recursos utilizados
- 5) Resultados obtidos, apresentando-os numa tabela e fazendo a sua análise e discussão
- 6) Conclusões, incluindo aspetos a melhorar no futuro
- Bibliografia utilizada

O resumo deve indicar os dados e os métodos utilizados, as opções mais relevantes e as principais conclusões. No caso de grupos com mais do que 1 elemento, o último parágrafo do resumo deve indicar a percentagem de contribuição de cada elemento para o trabalho, juntamente com uma pequena explicação. Por exemplo: Maria: 40%, Pedro: 30%, Manuel: 30%.

*Observação.* O relatório não deverá ter mais do que 5 tabelas com resultados. O resultado de cada uma das experiências poderá ser apresentado, juntamente com uma breve descrição das condições da experiência. Por exemplo:

|                                                                                              | Acc.  | Prec. | ... |
|----------------------------------------------------------------------------------------------|-------|-------|-----|
| Baseline (TextBlob)                                                                          | 0.660 |       |     |
| ...                                                                                          |       |       |     |
| SVM, C=1.0, tokenization, lowercase, stopwords, negation, POS, lemmatization, spell correction, 10K features, ... | 0.956 | ...   |     |

### Código

O grupo deverá enviar todo o código que permita replicar os resultados obtidos. No caso das experiências com modelos baseados em instruções, deverá também incluir as instruções (*prompts*) utilizadas, que poderá incluir como anexos no relatório.

### Apresentação

O grupo deverá preparar uma apresentação em formato `pdf` ou `pptx` para fazer **presencialmente** durante a aula. A apresentação deverá ser projetada para um máximo de **8 minutos** (tempos acima de 8 minutos serão penalizados), devendo cobrir os seguintes aspetos: descrição dos dados, abordagens e opções mais relevantes, resultados obtidos e propostas para trabalho futuro que poderiam levar a melhores resultados.

Os ficheiros da apresentação devem ser submetidos até às 17h00 do dia das apresentações através do link: https://t.ly/HQo5v. O grupo deve criar a pasta `GrupoXX`, em que `XX` é o número do grupo, e colocar os respetivos ficheiros dentro dessa pasta.

### Avaliação

Embora o código tenha obrigatoriamente de ser entregue, a avaliação do trabalho incidirá sobre o relatório e a apresentação, sendo o código usado apenas para clarificação. Na avaliação serão tidos em conta os seguintes critérios:

1. **(8v)** Correção e multiplicidade de soluções propostas
2. **(8v)** Estrutura e conteúdo do relatório
3. **(4v)** Apresentação (é obrigatória).

Eventuais atrasos na entrega do relatório ou apresentação serão penalizados.

---

## Política em caso de fraude

Os alunos podem partilhar e/ou trocar ideias entre si sobre os trabalhos e/ou resolução dos mesmos. No entanto, o trabalho entregue deve corresponder ao esforço individual de cada grupo. São consideradas fraudes as seguintes situações:

- Trabalho parcialmente copiado
- Facilitar a cópia através da partilha de ficheiros
- Utilizar material alheio sem referir a sua fonte.

Em caso de deteção de algum tipo de fraude, os trabalhos em questão não serão avaliados, sendo enviados à Comissão Pedagógica ou ao Conselho Pedagógico, consoante a gravidade da situação, que decidirão a sanção a aplicar aos alunos envolvidos. Serão utilizadas as ferramentas *Moss* e *SafeAssign* para detecção automática de cópias.

Recorda-se ainda que o Anexo I do Código de Conduta Académica, publicado a 25 de Janeiro de 2016 em Diário da República, 2ª Série, nº 16, indica no seu ponto 2 que:

> Quando um trabalho ou outro elemento de avaliação apresentar um nível de coincidência elevado com outros trabalhos (percentagem de coincidência com outras fontes reportada no relatório que o referido software produz), cabe ao docente da UC, orientador ou a qualquer elemento do júri, após a análise qualitativa desse relatório, e em caso de se confirmar a suspeita de plágio, desencadear o respetivo procedimento disciplinar, de acordo com o Regulamento Disciplinar de Discentes do ISCTE - Instituto Universitário de Lisboa, aprovado pela deliberação n.o 2246/2010, de 6 de dezembro.

O ponto 2.1 desse mesmo anexo indica ainda que:

> No âmbito do Regulamento Disciplinar de Discentes do ISCTE-IUL, são definidas as sanções disciplinares aplicáveis e os seus efeitos, podendo estas variar entre a advertência e a interdição da frequência de atividades escolares no ISCTE-IUL até cinco anos.
