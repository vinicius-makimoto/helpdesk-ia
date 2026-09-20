#Etapa 1: importar as bibliotecas
import joblib #Biblioteca utilizada para salvar e carregar modelos de Machine Learning
from pathlib import Path #Biblioteca utilizada para trabalhar com caminhos de arquivos e pastas
import pandas as pd #Biblioteca utilizada para trabalhar com tabelas e DataFrames
from sklearn.compose import ColumnTransformer #Permite aplicar tratamentos diferentes em diferentes colunas
from sklearn.feature_extraction.text import TfidfVectorizer #Transforma o texto dos chamados em características numéricas
from sklearn.preprocessing import OneHotEncoder #Transforma categoria e prioridade em características numéricas
from sklearn.pipeline import Pipeline #Agrupa o pré-processamento e o modelo em uma única estrutura
from sklearn.linear_model import LogisticRegression #Modelo utilizado para classificar o setor responsável
from sklearn.model_selection import train_test_split #Ferramenta utilizada para separar treinamento e teste
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #Métricas utilizadas para avaliar o modelo

#Etapa 2: localizar a base de dados
caminho_base = Path(__file__).parent / "data" / "chamados_v2.csv" #Localiza o arquivo chamados_v2.csv dentro da pasta data

#Etapa 3: carregar a base de dados
df = pd.read_csv(caminho_base) #Carrega o CSV e transforma os dados em um DataFrame

#Etapa 4: definir os dados de entrada e a informação que será prevista
X = df[["texto", "categoria", "prioridade"]] #Utiliza texto, categoria e prioridade como informações de entrada
y = df["setor"] #Define o setor responsável como informação que o modelo deverá aprender a prever

#Etapa 5: separar os dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, #Dados de entrada que serão divididos
    y, #Setores correspondentes aos chamados
    test_size=0.20, #Reserva 20% dos registros para teste
    random_state=42, #Mantém a mesma divisão sempre que o código for executado
    stratify=y #Mantém aproximadamente a proporção dos setores nos conjuntos
)

print("\nQuantidade para treinamento:")
print(len(X_train))

print("\nQuantidade para teste:")
print(len(X_test))

print("\nSetores no treinamento:")
print(y_train.value_counts())

print("\nSetores no teste:")
print(y_test.value_counts())

#Etapa 6: configurar o processamento das informações
preprocessador = ColumnTransformer(
    transformers=[
        ("texto", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True), "texto"), #Transforma o texto utilizando TF-IDF
        ("categoria", OneHotEncoder(handle_unknown="ignore"), ["categoria"]), #Transforma as categorias em valores numéricos
        ("prioridade", OneHotEncoder(handle_unknown="ignore"), ["prioridade"]) #Transforma as prioridades em valores numéricos
    ]
)

#Etapa 7: criar o pipeline do modelo
modelo_setor = Pipeline(
    steps=[
        ("preprocessador", preprocessador), #Executa automaticamente o tratamento das informações de entrada
        ("classificador", LogisticRegression(max_iter=1000, random_state=42)) #Classifica o setor utilizando as características processadas
    ]
)

#Etapa 8: treinar o modelo
modelo_setor.fit(X_train, y_train) #Aprende a relação entre texto, categoria, prioridade e setor responsável

#Etapa 9: realizar previsões
y_pred = modelo_setor.predict(X_test) #Solicita ao modelo que determine o setor dos chamados reservados para teste

#Etapa 10: calcular a acurácia
acuracia = accuracy_score(y_test, y_pred) #Calcula a porcentagem total de setores previstos corretamente

print("\nAcurácia do modelo de setor:")
print(f"{acuracia:.2%}")

#Etapa 11: gerar o relatório de classificação
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred)) #Exibe precision, recall e F1-score de cada setor

#Etapa 12: gerar a matriz de confusão
classes = modelo_setor.named_steps["classificador"].classes_ #Obtém a ordem das classes utilizada pelo classificador
matriz = confusion_matrix(y_test, y_pred, labels=classes) #Compara os setores verdadeiros com os setores previstos
matriz_df = pd.DataFrame(matriz, index=classes, columns=classes) #Transforma a matriz em uma tabela para facilitar a leitura

print("\nMatriz de confusão:")
print(matriz_df)

#Etapa 13: criar chamados para testar o modelo manualmente
novos_chamados = pd.DataFrame([
    {
        "texto": "Meu computador não está exibindo imagem na tela.",
        "categoria": "Hardware",
        "prioridade": "Média"
    },
    {
        "texto": "A conexão caiu para várias pessoas do escritório.",
        "categoria": "Rede",
        "prioridade": "Alta"
    },
    {
        "texto": "O sistema corporativo apresenta erro ao gerar um relatório.",
        "categoria": "Software",
        "prioridade": "Média"
    },
    {
        "texto": "Recebi uma mensagem suspeita solicitando minha senha.",
        "categoria": "Segurança",
        "prioridade": "Média"
    }
]) #Cria exemplos externos contendo as mesmas informações esperadas pelo modelo

#Etapa 14: realizar previsões dos novos chamados
previsoes = modelo_setor.predict(novos_chamados) #Utiliza todo o pipeline para processar os dados e prever os setores

print("\nTestes com chamados novos:")
for chamado, previsao in zip(novos_chamados["texto"], previsoes): #Percorre cada chamado junto com o setor previsto
    print(f"\nChamado: {chamado}")
    print(f"Setor previsto: {previsao}")

#Etapa 15: criar o diretório dos modelos
caminho_modelos = Path(__file__).parent / "models" #Define o caminho da pasta onde os modelos serão armazenados
caminho_modelos.mkdir(exist_ok=True) #Cria a pasta models caso ela ainda não exista

#Etapa 16: salvar o pipeline do modelo de setor
joblib.dump(modelo_setor, caminho_modelos / "modelo_setor_v1.pkl") #Salva o pipeline completo contendo pré-processamento e classificação do setor

print("\nModelo de setor salvo com sucesso.")