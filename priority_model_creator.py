#Etapa 1: importar as bibliotecas
from pathlib import Path #Biblioteca utilizada para trabalhar com caminhos de arquivos e pastas
import joblib #Biblioteca utilizada para salvar o modelo treinado
import pandas as pd #Biblioteca utilizada para trabalhar com tabelas e DataFrames
from sklearn.compose import ColumnTransformer #Permite aplicar diferentes tratamentos nas colunas
from sklearn.feature_extraction.text import TfidfVectorizer #Transforma o texto em características numéricas utilizando TF-IDF
from sklearn.preprocessing import OneHotEncoder #Transforma a categoria em características numéricas
from sklearn.pipeline import Pipeline #Agrupa o pré-processamento e o modelo em uma única estrutura
from sklearn.linear_model import LogisticRegression #Modelo utilizado para classificar a prioridade
from sklearn.model_selection import train_test_split #Ferramenta utilizada para separar treinamento e teste
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #Métricas utilizadas para avaliar o modelo

#Etapa 2: localizar a base de dados
caminho_base = Path(__file__).parent / "data" / "chamados_v2.csv" #Localiza o arquivo chamados_v2.csv dentro da pasta data

#Etapa 3: carregar a base de dados
df = pd.read_csv(caminho_base) #Carrega o CSV e transforma os dados em um DataFrame

#Etapa 4: definir as informações utilizadas pelo modelo
X = df[["texto", "categoria"]] #Utiliza o texto do chamado e sua categoria como informações de entrada
y = df["prioridade"] #Define a prioridade como informação que o modelo deverá aprender a prever

#Etapa 5: separar os dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, #Dados que serão utilizados como entrada
    y, #Prioridades correspondentes aos chamados
    test_size=0.20, #Reserva 20% da base para teste
    random_state=42, #Mantém a mesma divisão quando o código for executado novamente
    stratify=y #Mantém a proporção das prioridades nos conjuntos
)

print("\nQuantidade para treinamento:")
print(len(X_train))

print("\nQuantidade para teste:")
print(len(X_test))

print("\nPrioridades no treinamento:")
print(y_train.value_counts())

print("\nPrioridades no teste:")
print(y_test.value_counts())

#Etapa 6: configurar o processamento das informações
preprocessador_prioridade = ColumnTransformer(
    transformers=[
        ("texto", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True), "texto"), #Transforma o texto utilizando TF-IDF
        ("categoria", OneHotEncoder(handle_unknown="ignore"), ["categoria"]) #Transforma a categoria em uma representação numérica
    ]
)

#Etapa 7: criar o pipeline do modelo de prioridade
modelo_prioridade = Pipeline(
    steps=[
        ("preprocessador", preprocessador_prioridade), #Executa o processamento do texto e da categoria
        ("classificador", LogisticRegression(max_iter=1000, random_state=42)) #Classifica o chamado entre Baixa, Média, Alta ou Crítica
    ]
)

#Etapa 8: treinar o modelo
modelo_prioridade.fit(X_train, y_train) #Treina o pipeline utilizando os chamados e suas prioridades corretas

#Etapa 9: realizar previsões no conjunto de teste
y_pred = modelo_prioridade.predict(X_test) #Solicita ao modelo que determine as prioridades dos chamados reservados para teste

#Etapa 10: calcular a acurácia
acuracia = accuracy_score(y_test, y_pred) #Calcula a porcentagem de prioridades classificadas corretamente

print("\nAcurácia do modelo de prioridade:")
print(f"{acuracia:.2%}")

#Etapa 11: gerar o relatório de classificação
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred)) #Exibe precision, recall e F1-score de cada prioridade

#Etapa 12: gerar a matriz de confusão
classes = modelo_prioridade.named_steps["classificador"].classes_ #Obtém a ordem das prioridades utilizada pelo classificador
matriz = confusion_matrix(y_test, y_pred, labels=classes) #Compara as prioridades verdadeiras com as previstas
matriz_df = pd.DataFrame(matriz, index=classes, columns=classes) #Transforma a matriz em uma tabela

print("\nMatriz de confusão:")
print(matriz_df)

#Etapa 13: criar chamados externos para testar a generalização
novos_chamados = pd.DataFrame([
    {
        "texto": "Meu mouse parou de funcionar, mas tenho outro disponível e consigo continuar trabalhando.",
        "categoria": "Hardware"
    },
    {
        "texto": "Não consigo acessar minha conta mesmo digitando a senha correta.",
        "categoria": "Acesso"
    },
    {
        "texto": "A internet está caindo várias vezes e está afetando todo o departamento.",
        "categoria": "Rede"
    },
    {
        "texto": "Recebi um e-mail dizendo que minha conta será bloqueada e pedindo para informar minha senha.",
        "categoria": "Segurança"
    },
    {
        "texto": "O antivírus encontrou um programa desconhecido executando no meu computador.",
        "categoria": "Segurança"
    },
    {
        "texto": "A VPN parou de funcionar e toda a equipe em home office não consegue acessar os sistemas internos.",
        "categoria": "Rede"
    }
]) #Cria chamados escritos de maneira diferente dos exemplos utilizados durante o treinamento

#Etapa 14: prever a prioridade dos chamados externos
previsoes = modelo_prioridade.predict(novos_chamados) #Envia os chamados externos para o pipeline de prioridade

print("\nTestes com chamados novos:")
for chamado, previsao in zip(novos_chamados["texto"], previsoes): #Percorre cada chamado junto com sua prioridade prevista
    print(f"\nChamado: {chamado}")
    print(f"Prioridade prevista: {previsao}")

#Etapa 15: localizar a pasta dos modelos
caminho_modelos = Path(__file__).parent / "models" #Define o diretório onde o modelo será armazenado
caminho_modelos.mkdir(exist_ok=True) #Cria a pasta models caso ela ainda não exista

#Etapa 16: salvar o novo pipeline de prioridade
joblib.dump(modelo_prioridade, caminho_modelos / "modelo_prioridade_v1.pkl") #Substitui o modelo anterior pelo novo pipeline de prioridade

print("\nModelo de prioridade salvo com sucesso.")