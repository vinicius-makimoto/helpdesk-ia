#Etapa 1: importar as bibliotecas
import joblib #Biblioteca utilizada para salvar e carregar modelos de Machine Learning
from pathlib import Path #Biblioteca para trabalhar com caminhos de arquivos e pastas
import random #Biblioteca utilizada para realizar seleções aleatórias de forma controlada
import pandas as pd #Biblioteca utilizada para trabalhar com tabelas e DataFrames
from sklearn.feature_extraction.text import TfidfVectorizer #Converte textos em valores numéricos utilizando TF-IDF
from sklearn.linear_model import LogisticRegression #Modelo de Machine Learning utilizado para classificação
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #Métricas utilizadas para avaliar o modelo

#Etapa 2: localizar a base de dados
caminho_base = Path(__file__).parent / "data" / "chamados_v2.csv" #Localiza o CSV dentro da pasta data a partir da pasta atual do projeto

#Etapa 3: carregar a base de dados
df = pd.read_csv(caminho_base) #Carrega o arquivo CSV e transforma os dados em um DataFrame

#Etapa 4: visualizar informações da base
print("\nPrimeiros registros:")
print(df.head()) #Exibe os cinco primeiros registros da base

print("\nQuantidade de registros:")
print(len(df)) #Exibe a quantidade total de chamados

print("\nCategorias:")
print(df["categoria"].value_counts()) #Conta quantos registros existem em cada categoria

print("\nPrioridades:")
print(df["prioridade"].value_counts()) #Conta quantos registros existem em cada prioridade

print("\nSetores:")
print(df["setor"].value_counts()) #Conta quantos registros existem em cada setor

print("\nQuantidade de grupos:")
print(df["grupo"].nunique()) #Exibe quantos grupos conceituais diferentes existem na base

#Etapa 5: separar os grupos de treinamento, validação e teste
random.seed(42) #Define uma semente para que a seleção dos grupos seja sempre reproduzível
grupos_validacao = [] #Lista que armazenará um grupo de cada categoria para validação
grupos_teste = [] #Lista que armazenará um grupo de cada categoria para o teste final

for categoria in sorted(df["categoria"].unique()): #Percorre cada categoria existente na base
    grupos_categoria = sorted(df[df["categoria"] == categoria]["grupo"].unique().tolist()) #Busca os grupos pertencentes somente à categoria atual
    random.shuffle(grupos_categoria) #Embaralha os grupos para evitar uma escolha baseada na ordem do CSV
    grupos_validacao.append(grupos_categoria[0]) #Reserva o primeiro grupo embaralhado exclusivamente para validação
    grupos_teste.append(grupos_categoria[1]) #Reserva o segundo grupo embaralhado exclusivamente para o teste final

print("\nGrupos reservados para validação:")
print(grupos_validacao)

print("\nGrupos reservados para teste final:")
print(grupos_teste)

#Etapa 6: criar os conjuntos de treinamento, validação e teste
df_validacao = df[df["grupo"].isin(grupos_validacao)].copy() #Seleciona somente os registros pertencentes aos grupos de validação
df_test = df[df["grupo"].isin(grupos_teste)].copy() #Seleciona somente os registros pertencentes aos grupos do teste final
df_train = df[~df["grupo"].isin(grupos_validacao + grupos_teste)].copy() #Utiliza os demais grupos exclusivamente para treinamento

#Etapa 7: separar os textos e categorias de cada conjunto
X_train = df_train["texto"] #Textos que serão utilizados para treinar o modelo
y_train = df_train["categoria"] #Categorias corretas utilizadas durante o treinamento

X_validacao = df_validacao["texto"] #Textos utilizados para avaliar os ajustes realizados no modelo
y_validacao = df_validacao["categoria"] #Categorias corretas utilizadas na validação

X_test = df_test["texto"] #Textos reservados para a avaliação final do modelo
y_test = df_test["categoria"] #Categorias corretas reservadas para a avaliação final

print("\nQuantidade para treinamento:")
print(len(X_train))

print("\nQuantidade para validação:")
print(len(X_validacao))

print("\nQuantidade reservada para teste final:")
print(len(X_test))

#Etapa 8: criar o vetorizador TF-IDF
vetorizador = TfidfVectorizer(
    lowercase=True, #Converte todas as letras para minúsculas para evitar palavras iguais sendo tratadas de formas diferentes
    ngram_range=(1, 2), #Analisa palavras individuais e combinações de duas palavras consecutivas
    min_df=2, #Ignora termos que aparecem em menos de dois chamados do treinamento
    max_df=0.95, #Ignora termos presentes em mais de 95% dos chamados por terem pouco poder de diferenciação
    sublinear_tf=True #Reduz a influência de palavras repetidas muitas vezes dentro de um mesmo texto
)

#Etapa 9: aprender o vocabulário utilizando somente os dados de treinamento
X_train_tfidf = vetorizador.fit_transform(X_train) #Aprende o vocabulário do treinamento e transforma os textos em números
X_validacao_tfidf = vetorizador.transform(X_validacao) #Transforma a validação usando somente o vocabulário aprendido no treinamento
X_test_tfidf = vetorizador.transform(X_test) #Transforma o teste final sem ensinar novas palavras ao vetorizador

#Etapa 10: visualizar informações do TF-IDF
print("\nFormato dos dados de treinamento após TF-IDF:")
print(X_train_tfidf.shape) #Exibe quantidade de chamados e características criadas pelo TF-IDF

print("\nAlgumas palavras e termos aprendidos:")
print(vetorizador.get_feature_names_out()[:30]) #Exibe uma pequena amostra do vocabulário aprendido

#Etapa 11: criar o modelo de classificação de categoria
modelo_categoria = LogisticRegression(max_iter=1000, random_state=42) #Cria o classificador e permite até 1000 iterações durante o treinamento

#Etapa 12: treinar o modelo
modelo_categoria.fit(X_train_tfidf, y_train) #Ensina o modelo utilizando os textos transformados e suas categorias corretas

#Etapa 13: realizar previsões utilizando o conjunto de validação
y_pred_validacao = modelo_categoria.predict(X_validacao_tfidf) #Solicita ao modelo que classifique chamados que não participaram do treinamento

#Etapa 14: calcular a acurácia da validação
acuracia_validacao = accuracy_score(y_validacao, y_pred_validacao) #Calcula a porcentagem total de previsões corretas

print("\nAcurácia da validação:")
print(f"{acuracia_validacao:.2%}")

#Etapa 15: gerar o relatório de classificação da validação
print("\nRelatório de classificação da validação:")
print(classification_report(y_validacao, y_pred_validacao)) #Exibe precision, recall, F1-score e quantidade de exemplos por categoria

#Etapa 16: gerar a matriz de confusão da validação
matriz = confusion_matrix(y_validacao, y_pred_validacao, labels=modelo_categoria.classes_) #Compara as categorias verdadeiras com as categorias previstas pelo modelo
matriz_df = pd.DataFrame(matriz, index=modelo_categoria.classes_, columns=modelo_categoria.classes_) #Transforma a matriz em um DataFrame para facilitar a leitura

print("\nMatriz de confusão da validação:")
print(matriz_df)

#Etapa 17: testar o modelo com chamados escritos manualmente
novos_chamados = [
    "Meu notebook liga normalmente, mas não aparece nada na tela.",
    "Não estou conseguindo entrar na minha conta mesmo usando a senha correta.",
    "Desde cedo a conexão fica caindo e voltando.",
    "Recebi uma mensagem estranha pedindo para confirmar meus dados.",
    "O programa usado pelo financeiro fecha quando tentamos emitir o relatório.",
    "Mandei imprimir um documento, mas nada saiu.",
    "Apareceu um programa no meu computador que eu nunca instalei."
] #Chamados externos utilizados para observar o comportamento do modelo em frases diferentes da base

novos_chamados_tfidf = vetorizador.transform(novos_chamados) #Transforma os novos chamados utilizando o mesmo TF-IDF aprendido no treinamento
previsoes = modelo_categoria.predict(novos_chamados_tfidf) #Realiza a previsão da categoria de cada novo chamado

print("\nTestes com chamados novos:")
for chamado, previsao in zip(novos_chamados, previsoes): #Percorre cada chamado junto com a categoria prevista pelo modelo
    print(f"\nChamado: {chamado}")
    print(f"Categoria prevista: {previsao}")

#Etapa 18: criar o diretório dos modelos
caminho_modelos = Path(__file__).parent / "models" #Define o caminho da pasta onde os modelos serão armazenados
caminho_modelos.mkdir(exist_ok=True) #Cria a pasta models caso ela ainda não exista

#Etapa 19: salvar o modelo e o vetorizador de categoria
joblib.dump(modelo_categoria, caminho_modelos / "modelo_categoria_v1.pkl") #Salva o modelo treinado responsável pela categoria
joblib.dump(vetorizador, caminho_modelos / "vetorizador_categoria_v1.pkl") #Salva o TF-IDF utilizado pelo modelo de categoria

print("\nModelo de categoria salvo com sucesso.")