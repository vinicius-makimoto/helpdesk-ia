#Etapa 1: importar as bibliotecas e limitar os recursos utilizados pelas bibliotecas numéricas
import os #Biblioteca utilizada para configurar variáveis de ambiente do servidor

#Limita o OpenBLAS para utilizar apenas uma thread no servidor
os.environ["OPENBLAS_NUM_THREADS"] = "1"

#Limita outras bibliotecas numéricas para evitar criação excessiva de threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path #Biblioteca utilizada para trabalhar com caminhos de arquivos e pastas
import joblib #Biblioteca utilizada para carregar os modelos de Machine Learning salvos
import pandas as pd #Biblioteca utilizada para criar os dados enviados aos modelos
from flask import Flask, render_template, request #Biblioteca Flask utilizada para criar a aplicação web, carregar páginas HTML e receber formulários

#Etapa 2: criar a aplicação Flask
app = Flask(__name__) #Cria a aplicação Flask

#Etapa 3: localizar a pasta dos modelos
caminho_modelos = Path(__file__).parent / "models" #Localiza a pasta models a partir da pasta onde está o app.py

#Etapa 4: carregar o modelo e o vetorizador de categoria
modelo_categoria = joblib.load(caminho_modelos / "modelo_categoria_v1.pkl") #Carrega o modelo responsável por prever a categoria
vetorizador_categoria = joblib.load(caminho_modelos / "vetorizador_categoria_v1.pkl") #Carrega o TF-IDF utilizado durante o treinamento da categoria

#Etapa 5: carregar o modelo de prioridade
modelo_prioridade = joblib.load(caminho_modelos / "modelo_prioridade_v1.pkl") #Carrega o pipeline responsável por prever a prioridade

#Etapa 6: carregar o modelo de setor
modelo_setor = joblib.load(caminho_modelos / "modelo_setor_v1.pkl") #Carrega o pipeline completo responsável por prever o setor

#Etapa 7: criar a rota principal
@app.route("/")
def inicio():
    return render_template("index.html") #Carrega a página inicial da aplicação

#Etapa 8: criar a rota responsável pela análise do chamado
@app.route("/analisar", methods=["POST"])
def analisar():
    texto_chamado = request.form.get("chamado", "").strip() #Recebe o texto digitado no formulário e remove espaços desnecessários

    if not texto_chamado: #Verifica se o usuário tentou enviar um chamado vazio
        return render_template("index.html", erro="Digite uma descrição para o chamado.")

    #Etapa 9: prever a categoria
    texto_categoria = vetorizador_categoria.transform([texto_chamado]) #Transforma o chamado utilizando o mesmo TF-IDF usado no treinamento da categoria
    categoria = modelo_categoria.predict(texto_categoria)[0] #Realiza a previsão e obtém a categoria retornada pelo modelo

    #Etapa 10: preparar os dados utilizados pelo modelo de prioridade
    dados_prioridade = pd.DataFrame([{
        "texto": texto_chamado,
        "categoria": categoria
    }]) #Cria um DataFrame contendo o texto do chamado e a categoria prevista

    #Etapa 11: prever a prioridade
    prioridade = modelo_prioridade.predict(dados_prioridade)[0] #Envia texto e categoria para o pipeline e obtém a prioridade prevista

    #Etapa 12: preparar os dados utilizados pelo modelo de setor
    dados_setor = pd.DataFrame([{
        "texto": texto_chamado,
        "categoria": categoria,
        "prioridade": prioridade
    }]) #Cria um DataFrame contendo texto, categoria prevista e prioridade prevista

    #Etapa 13: prever o setor responsável
    setor = modelo_setor.predict(dados_setor)[0] #Envia os dados ao pipeline e obtém o setor previsto

    #Etapa 14: enviar os resultados para a interface
    return render_template(
        "index.html",
        chamado=texto_chamado,
        categoria=categoria,
        prioridade=prioridade,
        setor=setor
    ) #Retorna a página HTML contendo o resultado completo da análise

#Etapa 15: iniciar o servidor Flask durante o desenvolvimento local
if __name__ == "__main__":
    app.run(debug=True) #Inicia o servidor local do Flask com o modo de depuração ativado