# Etapa 1: importar bibliotecas e limitar recursos numéricos

import os
import re

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from datetime import datetime
from pathlib import Path
import secrets

import joblib
import pandas as pd
import pymysql
from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    session,
    url_for
)

from werkzeug.security import (
    check_password_hash,
    generate_password_hash
)


from pymysql.cursors import DictCursor

# Etapa 2: criar a aplicação Flask

app = Flask(__name__)

# Etapa 2.0: configurar a chave de sessão sem expor segredo no código

chave_secreta_flask = os.getenv(
    "FLASK_SECRET_KEY",
    ""
).strip()

if chave_secreta_flask:
    app.config["SECRET_KEY"] = chave_secreta_flask


# Etapa 2.1: definir erro seguro de conexão com o banco

class ErroBancoDados(Exception):
    """
    Representa uma falha técnica ao acessar o MariaDB.

    A mensagem técnica original não deve ser exibida
    diretamente ao usuário final.
    """
    pass

# Etapa 2.1.1: tratar falha de persistência sem expor dados técnicos

@app.errorhandler(ErroBancoDados)
def tratar_erro_banco(erro):
    mensagem_banco = (
        "Não foi possível registrar o chamado "
        "no momento."
    )

    texto_chamado = request.form.get(
        "chamado",
        ""
    ).strip()

    return render_template(
        "index.html",
        chamado=texto_chamado,
        erro=mensagem_banco,
        message=mensagem_banco,
        decision="SERVICE_UNAVAILABLE",
        reason_code="DATABASE_UNAVAILABLE",
        confidence_action=None,
        confidence_level=None,
        confidence_message=None,
        confianca_categoria=None,
        confianca_prioridade=None,
        confianca_setor=None,
        confianca_final=None,
        triage_context={
            "urgency_signal": False,
            "business_impact": False,
            "urgency_reason": None,
            "requested_urgency": None
        },
        priority_predicted=None,
        priority_source="NONE",
        classification_status="REVIEW_ONLY",
        classification_available=False,
        classification_complete=False,
        needs_human_review=False,
        protocolo=None,
        chamado_registrado=False
    ), 503

# Etapa 2.2: montar a configuração segura do banco

def obter_configuracao_banco():
    """
    Lê as credenciais somente por variáveis de ambiente.

    Nenhuma senha deve ser declarada diretamente
    no código-fonte.
    """

    nome_banco = os.getenv("DB_NAME", "").strip()
    usuario_banco = os.getenv("DB_USER", "").strip()
    senha_banco = os.getenv("DB_PASSWORD", "")
    host_banco = os.getenv("DB_HOST", "localhost").strip()
    socket_banco = os.getenv("DB_UNIX_SOCKET", "").strip()

    try:
        porta_banco = int(
            os.getenv("DB_PORT", "3306")
        )
    except ValueError as erro:
        raise ErroBancoDados(
            "A porta configurada para o banco é inválida."
        ) from erro

    if not nome_banco:
        raise ErroBancoDados(
            "O nome do banco não foi configurado."
        )

    if not usuario_banco:
        raise ErroBancoDados(
            "O usuário do banco não foi configurado."
        )

    if not senha_banco:
        raise ErroBancoDados(
            "A senha do banco não foi configurada."
        )

    configuracao = {
        "user": usuario_banco,
        "password": senha_banco,
        "database": nome_banco,
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": False,
        "connect_timeout": 10,
        "read_timeout": 10,
        "write_timeout": 10
    }

    if socket_banco:
        configuracao["unix_socket"] = socket_banco
    else:
        configuracao["host"] = host_banco
        configuracao["port"] = porta_banco

    return configuracao

# Etapa 2.3: abrir conexão com o MariaDB

def obter_conexao_banco():
    """
    Abre uma nova conexão com o banco.

    A conexão deve ser fechada por quem a utiliza.
    """

    try:
        return pymysql.connect(
            **obter_configuracao_banco()
        )
    except pymysql.MySQLError as erro:
        raise ErroBancoDados(
            "Não foi possível conectar ao banco de dados."
        ) from erro

# Etapa 2.4: validar a conexão sem alterar dados

def verificar_conexao_banco():
    """
    Executa somente uma consulta de leitura.

    Não cria, altera ou remove registros.
    """

    conexao = None
    cursor = None

    try:
        conexao = obter_conexao_banco()
        cursor = conexao.cursor()

        cursor.execute(
            "SELECT 1 AS banco_disponivel"
        )

        resultado = cursor.fetchone()

        return resultado["banco_disponivel"] == 1

    except ErroBancoDados:
        raise

    except pymysql.MySQLError as erro:
        raise ErroBancoDados(
            "Não foi possível validar a conexão com o banco."
        ) from erro

    finally:
        if cursor is not None:
            cursor.close()

        if conexao is not None:
            conexao.close()

# Etapa 2.5: gerar protocolo único para o chamado

def gerar_protocolo():
    """
    Gera um protocolo público para acompanhamento do chamado.

    A unicidade também é protegida pela chave UNIQUE
    existente na coluna protocolo do MariaDB.
    """

    data_atual = datetime.now().strftime(
        "%Y%m%d"
    )

    codigo_aleatorio = secrets.token_hex(
        4
    ).upper()

    return (
        f"HD-{data_atual}-{codigo_aleatorio}"
    )

# Etapa 2.6: registrar chamado e histórico no banco

def registrar_chamado_no_banco(dados_chamado):
    """
    Salva o chamado e registra o evento inicial de histórico.

    As duas operações fazem parte da mesma transação.
    Se qualquer etapa falhar, o rollback impede
    registros incompletos.
    """

    conexao = None
    cursor = None

    try:
        protocolo = gerar_protocolo()

        conexao = obter_conexao_banco()
        cursor = conexao.cursor()

        query_chamado = """
            INSERT INTO chamados (
                protocolo,
                descricao,
                decision,
                reason_code,
                categoria,
                prioridade,
                setor,
                confianca_categoria,
                confianca_prioridade,
                confianca_setor,
                confianca_final,
                confidence_action,
                confidence_level,
                classification_status,
                classification_available,
                classification_complete,
                needs_human_review,
                priority_predicted,
                priority_source,
                urgency_signal,
                business_impact,
                urgency_reason,
                requested_urgency,
                status
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
        """

        valores_chamado = (
            protocolo,
            dados_chamado["descricao"],
            dados_chamado["decision"],
            dados_chamado.get("reason_code"),
            dados_chamado.get("categoria"),
            dados_chamado.get("prioridade"),
            dados_chamado.get("setor"),
            dados_chamado.get("confianca_categoria"),
            dados_chamado.get("confianca_prioridade"),
            dados_chamado.get("confianca_setor"),
            dados_chamado.get("confianca_final"),
            dados_chamado.get("confidence_action"),
            dados_chamado.get("confidence_level"),
            dados_chamado.get(
                "classification_status",
                "REVIEW_ONLY"
            ),
            int(
                bool(
                    dados_chamado.get(
                        "classification_available",
                        False
                    )
                )
            ),
            int(
                bool(
                    dados_chamado.get(
                        "classification_complete",
                        False
                    )
                )
            ),
            int(
                bool(
                    dados_chamado.get(
                        "needs_human_review",
                        False
                    )
                )
            ),
            dados_chamado.get("priority_predicted"),
            dados_chamado.get(
                "priority_source",
                "NONE"
            ),
            int(
                bool(
                    dados_chamado.get(
                        "urgency_signal",
                        False
                    )
                )
            ),
            int(
                bool(
                    dados_chamado.get(
                        "business_impact",
                        False
                    )
                )
            ),
            dados_chamado.get("urgency_reason"),
            dados_chamado.get(
                "requested_urgency"
            ),
            dados_chamado.get("status", "NOVO")
        )

        cursor.execute(
            query_chamado,
            valores_chamado
        )

        chamado_id = cursor.lastrowid

        if not chamado_id:
            raise ErroBancoDados(
                "O banco não retornou o identificador "
                "do chamado criado."
            )

        query_historico = """
            INSERT INTO historico_chamado (
                chamado_id,
                usuario_id,
                acao,
                campo_alterado,
                valor_anterior,
                valor_novo,
                observacao
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s
            )
        """

        valores_historico = (
            chamado_id,
            None,
            "CRIADO",
            "status",
            None,
            dados_chamado.get("status", "NOVO"),
            (
                "Chamado registrado pela triagem "
                "automática."
            )
        )

        cursor.execute(
            query_historico,
            valores_historico
        )

        conexao.commit()

        return {
            "chamado_id": chamado_id,
            "protocolo": protocolo
        }

    except ErroBancoDados as erro:
        if conexao is not None:
            conexao.rollback()

        raise

    except (
        pymysql.MySQLError,
        KeyError,
        TypeError
    ) as erro:
        if conexao is not None:
            conexao.rollback()


        raise ErroBancoDados(
            "Não foi possível registrar o chamado "
            "no banco de dados."
        ) from erro

    finally:
        if cursor is not None:
            cursor.close()

        if conexao is not None:
            conexao.close()

# Etapa 2.7: definir erros seguros do cadastro e autenticação

class ErroCadastro(Exception):
    """
    Representa um erro de validação exibível ao usuário.

    Não deve incluir SQL, senha, hash ou detalhes internos.
    """
    pass


class EmailJaCadastrado(Exception):
    """
    Indica tentativa de cadastro com e-mail já existente.
    """
    pass


# Etapa 2.8: definir regras do cadastro

TAMANHO_MINIMO_SENHA = 10
TAMANHO_MAXIMO_SENHA = 128

PERFIS_AUTORIZADOS_DASHBOARD = {
    "ANALISTA",
    "ADMIN"
}


# Etapa 2.9: gerar token CSRF para os formulários

def obter_token_csrf():
    """
    Cria ou reutiliza um token vinculado à sessão Flask.

    O token deve ser enviado em formulários de login,
    cadastro e logout.
    """

    if not app.config.get("SECRET_KEY"):
        abort(503)

    token_csrf = session.get("csrf_token")

    if not token_csrf:
        token_csrf = secrets.token_urlsafe(32)
        session["csrf_token"] = token_csrf

    return token_csrf


# Etapa 2.10: validar token CSRF recebido pelo formulário

def validar_token_csrf():
    """
    Rejeita formulários sem token válido.
    """

    token_enviado = request.form.get(
        "csrf_token",
        ""
    )

    token_esperado = session.get(
        "csrf_token",
        ""
    )

    if (
        not token_enviado
        or not token_esperado
        or not secrets.compare_digest(
            token_enviado,
            token_esperado
        )
    ):
        abort(400)


# Etapa 2.11: validar os campos do cadastro

def validar_dados_cadastro(
    nome,
    sobrenome,
    email,
    senha
):
    """
    Valida apenas formato e integridade dos dados.

    A senha não é alterada nem registrada em logs.
    """

    padrao_nome = (
        r"[A-Za-zÀ-ÖØ-öø-ÿ' -]{2,120}"
    )

    padrao_email = (
        r"[^@\s]+@[^@\s]+\.[^@\s]+"
    )

    if not re.fullmatch(
        padrao_nome,
        nome
    ):
        raise ErroCadastro(
            "Digite um nome válido."
        )

    if not re.fullmatch(
        padrao_nome,
        sobrenome
    ):
        raise ErroCadastro(
            "Digite um sobrenome válido."
        )

    if (
        not re.fullmatch(
            padrao_email,
            email
        )
        or len(email) > 190
    ):
        raise ErroCadastro(
            "Digite um e-mail válido."
        )

    if (
        len(senha) < TAMANHO_MINIMO_SENHA
        or len(senha) > TAMANHO_MAXIMO_SENHA
    ):
        raise ErroCadastro(
            "A senha deve ter entre 10 e 128 caracteres."
        )


# Etapa 2.12: registrar usuário pendente no MariaDB

def registrar_usuario_pendente(
    nome,
    sobrenome,
    email,
    senha
):
    """
    Cria um usuário ainda não autorizado.

    A senha é transformada em hash antes do INSERT.
    A conta recebe ativo = 0 e não pode acessar
    o dashboard até liberação manual.
    """

    conexao = None
    cursor = None

    try:
        senha_hash = generate_password_hash(
            senha
        )

        conexao = obter_conexao_banco()
        cursor = conexao.cursor()

        cursor.execute(
            """
            SELECT id
            FROM usuarios_admin
            WHERE email = %s
            LIMIT 1
            """,
            (email,)
        )

        usuario_existente = cursor.fetchone()

        if usuario_existente is not None:
            raise EmailJaCadastrado()

        query_cadastro = """
            INSERT INTO usuarios_admin (
                nome,
                sobrenome,
                email,
                senha_hash,
                perfil,
                ativo
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
        """

        cursor.execute(
            query_cadastro,
            (
                nome,
                sobrenome,
                email,
                senha_hash,
                "ANALISTA",
                0
            )
        )

        usuario_id = cursor.lastrowid

        if not usuario_id:
            raise ErroBancoDados(
                "O banco não retornou o identificador "
                "do usuário criado."
            )

        conexao.commit()

        return {
            "usuario_id": usuario_id,
            "email": email
        }

    except EmailJaCadastrado:
        if conexao is not None:
            conexao.rollback()

        raise

    except pymysql.err.IntegrityError as erro:
        if conexao is not None:
            conexao.rollback()

        codigo_erro = (
            erro.args[0]
            if erro.args
            else None
        )

        if codigo_erro == 1062:
            raise EmailJaCadastrado() from erro

        raise ErroBancoDados(
            "Não foi possível concluir o cadastro."
        ) from erro

    except pymysql.MySQLError as erro:
        if conexao is not None:
            conexao.rollback()

        raise ErroBancoDados(
            "Não foi possível concluir o cadastro."
        ) from erro

    finally:
        if cursor is not None:
            cursor.close()

        if conexao is not None:
            conexao.close()


# Etapa 2.13: localizar usuário para autenticação

def obter_usuario_por_email(email):
    """
    Busca somente os campos necessários para autenticação.
    """

    conexao = None
    cursor = None

    try:
        conexao = obter_conexao_banco()
        cursor = conexao.cursor()

        cursor.execute(
            """
            SELECT
                id,
                nome,
                email,
                senha_hash,
                perfil,
                ativo
            FROM usuarios_admin
            WHERE email = %s
            LIMIT 1
            """,
            (email,)
        )

        return cursor.fetchone()

    except pymysql.MySQLError as erro:
        raise ErroBancoDados(
            "Não foi possível validar o acesso."
        ) from erro

    finally:
        if cursor is not None:
            cursor.close()

        if conexao is not None:
            conexao.close()

# Etapa 3: localizar a pasta dos modelos

caminho_modelos = Path(__file__).parent / "models"

# Etapa 4: carregar modelos e vetorizador

modelo_categoria = joblib.load(
    caminho_modelos / "modelo_categoria_v1.pkl"
)

vetorizador_categoria = joblib.load(
    caminho_modelos / "vetorizador_categoria_v1.pkl"
)

modelo_prioridade = joblib.load(
    caminho_modelos / "modelo_prioridade_v1.pkl"
)

modelo_setor = joblib.load(
    caminho_modelos / "modelo_setor_v1.pkl"
)

# Etapa 5: definir limites do guardrail

TAMANHO_MINIMO_CHAMADO = 12
TAMANHO_MAXIMO_CHAMADO = 1000

# Etapa 6: definir decisões do guardrail

DECISAO_ACEITAR = "ACEITAR"
DECISAO_REJEITAR = "REJEITAR"
DECISAO_ESCLARECER = "ESCLARECER"
DECISAO_ESCALONAR = "SECURITY_ESCALATION"

# Etapa 7: definir faixas de confiança

LIMIAR_CONFIANCA_NORMAL = 0.70
LIMIAR_CONFIANCA_ESCLARECIMENTO = 0.40

# Etapa 8: executar guardrails antes dos modelos

def avaliar_guardrail(texto_chamado):
    """
    Avalia o chamado antes de qualquer chamada aos modelos.

    Esta função não classifica categoria, prioridade ou setor.
    Ela somente verifica validade, segurança, escopo
    e necessidade de esclarecimento.
    """

    if not isinstance(texto_chamado, str):
        return {
            "decision": DECISAO_REJEITAR,
            "reason_code": "FIELD_INVALID",
            "message": "O conteúdo enviado é inválido."
        }

    texto_normalizado = texto_chamado.lower().strip()

    if not texto_normalizado:
        return {
            "decision": DECISAO_REJEITAR,
            "reason_code": "FIELD_MISSING",
            "message": "Digite uma descrição para o chamado."
        }

    if len(texto_normalizado) > TAMANHO_MAXIMO_CHAMADO:
        return {
            "decision": DECISAO_REJEITAR,
            "reason_code": "PAYLOAD_TOO_LARGE",
            "message": (
                "A descrição do chamado ultrapassa "
                "o limite permitido."
            )
        }

    padroes_injecao = [
        r"ignore\s+(as\s+)?instru[cç][oõ]es",
        r"ignore\s+(as\s+)?regras",
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"desconsidere\s+(as\s+)?instru[cç][oõ]es",
        r"desconsidere\s+(as\s+)?regras",
        r"revele\s+(o\s+)?prompt",
        r"mostre\s+(o\s+)?prompt",
        r"mostre\s+(as\s+)?suas\s+instru[cç][oõ]es",
        r"mostre\s+as\s+instru[cç][oõ]es\s+internas",
        r"prompt\s+injection"
    ]

    for padrao in padroes_injecao:
        if re.search(padrao, texto_normalizado):
            return {
                "decision": DECISAO_ESCALONAR,
                "reason_code": "PROMPT_INJECTION",
                "message": (
                    "Não foi possível processar esta solicitação."
                )
            }

    padroes_segredo = [
        (
            r"(revele|mostre|informe|envie|compartilhe)"
            r".{0,40}"
            r"(senha|token|api[\s_-]?key|chave secreta)"
        ),
        (
            r"(senha|token|api[\s_-]?key|chave secreta)"
            r".{0,40}"
            r"(revele|mostre|informe|envie|compartilhe)"
        )
    ]

    for padrao in padroes_segredo:
        if re.search(padrao, texto_normalizado):
            return {
                "decision": DECISAO_ESCALONAR,
                "reason_code": "CREDENTIAL_OR_SECRET_REQUEST",
                "message": (
                    "Não foi possível processar esta solicitação."
                )
            }

    padroes_fora_escopo = [
        r"\bdor\s+de\s+barriga\b",
        r"\bdor\s+de\s+cabe[cç]a\b",
        r"\bfebre\b",
        r"\btosse\b",
        r"\bdiarreia\b",
        r"\brem[eé]dio\b",
        r"\bsintomas?\b",
        r"\bm[eé]dico\b",
        r"\bconsulta\s+m[eé]dica\b",
        r"\bprevis[aã]o\s+do\s+tempo\b",
        r"\btemperatura\s+(de\s+)?hoje\b",
        r"\bvai\s+chover\b",
        r"\bclima\s+(de\s+hoje|amanh[aã])\b"
    ]

    for padrao in padroes_fora_escopo:
        if re.search(padrao, texto_normalizado):
            return {
                "decision": DECISAO_REJEITAR,
                "reason_code": "OUT_OF_SCOPE",
                "message": (
                    "Esta solicitação não corresponde "
                    "ao escopo do Helpdesk."
                )
            }

    if len(texto_normalizado) < TAMANHO_MINIMO_CHAMADO:
        return {
            "decision": DECISAO_ESCLARECER,
            "reason_code": "MISSING_CONTEXT",
            "message": (
                "Informe mais detalhes sobre o problema."
            )
        }

    frases_ambiguas = [
        "não funciona",
        "nao funciona",
        "deu erro",
        "está com problema",
        "esta com problema",
        "preciso de ajuda",
        "estou cansado",
        "estou cansada"
    ]

    if any(
        frase in texto_normalizado
        for frase in frases_ambiguas
    ):
        return {
            "decision": DECISAO_ESCLARECER,
            "reason_code": "AMBIGUOUS_REQUEST",
            "message": (
                "Para ajudar, informe qual sistema, equipamento, "
                "acesso ou serviço de TI está envolvido e qual "
                "problema ocorreu."
            )
        }

    return {
        "decision": DECISAO_ACEITAR,
        "reason_code": "ACCEPTED_IN_SCOPE",
        "message": "Chamado aceito para classificação."
    }

# Etapa 9: definir erro específico de confiança do modelo

class ErroConfiancaModelo(Exception):
    pass

# Etapa 10: calcular confiança e alinhar classes do modelo

def obter_confianca(modelo, dados, previsao):
    """
    Retorna a maior probabilidade produzida pelo modelo.

    A maior probabilidade deve corresponder à classe
    retornada por predict().
    """

    predict_proba = getattr(modelo, "predict_proba", None)

    if not callable(predict_proba):
        raise ErroConfiancaModelo(
            "O modelo não possui predict_proba()."
        )

    classes = getattr(modelo, "classes_", None)

    if classes is None:
        raise ErroConfiancaModelo(
            "O modelo não possui classes_."
        )

    try:
        probabilidades = predict_proba(dados)[0]
    except Exception as erro:
        raise ErroConfiancaModelo(
            "Falha ao executar predict_proba()."
        ) from erro

    if len(probabilidades) == 0:
        raise ErroConfiancaModelo(
            "O modelo não retornou probabilidades."
        )

    if len(classes) != len(probabilidades):
        raise ErroConfiancaModelo(
            "Classes e probabilidades estão desalinhadas."
        )

    try:
        probabilidades_numericas = [
            float(probabilidade)
            for probabilidade in probabilidades
        ]
    except Exception as erro:
        raise ErroConfiancaModelo(
            "As probabilidades retornadas são inválidas."
        ) from erro

    if any(
        probabilidade < 0.0 or probabilidade > 1.0
        for probabilidade in probabilidades_numericas
    ):
        raise ErroConfiancaModelo(
            "A probabilidade está fora do intervalo permitido."
        )

    indice_maior_probabilidade = max(
        range(len(probabilidades_numericas)),
        key=probabilidades_numericas.__getitem__
    )

    classe_da_probabilidade = classes[
        indice_maior_probabilidade
    ]

    if str(classe_da_probabilidade) != str(previsao):
        raise ErroConfiancaModelo(
            "A classe de predict() não corresponde "
            "à maior probabilidade."
        )

    return probabilidades_numericas[
        indice_maior_probabilidade
    ]

# Etapa 11: classificar confiança final

def classificar_nivel_confianca(confianca_final):
    """
    Classifica a confiança sem alterar a decisão de escopo.

    A decisão ACEITAR continua representando o escopo.
    A ação da confiança é informada separadamente.
    """

    if confianca_final >= LIMIAR_CONFIANCA_NORMAL:
        return {
            "nivel": "NORMAL",
            "decision": DECISAO_ACEITAR,
            "confidence_action": "NORMAL",
            "message": (
                "O chamado foi analisado normalmente."
            )
        }

    if confianca_final >= LIMIAR_CONFIANCA_ESCLARECIMENTO:
        return {
            "nivel": "CAUTELA",
            "decision": DECISAO_ACEITAR,
            "confidence_action": "CAUTION",
            "message": (
                "A análise é provisória. "
                "Confirme os dados do chamado."
            )
        }

    return {
        "nivel": "LOW_CONFIDENCE_REVIEW",
        "decision": DECISAO_ACEITAR,
        "confidence_action": "LOW_CONFIDENCE_REVIEW",
        "message": (
            "O chamado foi recebido, mas precisa de revisão manual "
            "antes da classificação definitiva."
        )
    }

# Etapa 12: criar a rota principal

@app.route("/")
def inicio():
    return render_template("index.html")

# Etapa 13: criar a rota de análise do chamado

@app.route("/analisar", methods=["POST"])
def analisar():
    texto_chamado = request.form.get(
        "chamado",
        ""
    ).strip()

    texto_normalizado = texto_chamado.lower()

    possui_urgencia = bool(
        re.search(
            r"\b(urg[êe]ncia|urgente|prioridade)\b",
            texto_normalizado
        )
    )

    possui_impacto_faturamento = bool(
        re.search(
            (
                r"\b(faturamento|financeiro|financeira|"
                r"receita|vendas|cobran[cç]a)\b"
            ),
            texto_normalizado
        )
    )

    triage_context = {
        "urgency_signal": (
            possui_urgencia or possui_impacto_faturamento
        ),
        "business_impact": possui_impacto_faturamento,
        "urgency_reason": (
            "BILLING_IMPACT"
            if possui_impacto_faturamento
            else (
                "USER_REQUESTED_URGENCY"
                if possui_urgencia
                else None
            )
        ),
        "requested_urgency": (
            "URGENTE"
            if possui_urgencia or possui_impacto_faturamento
            else None
        )
    }

    def obter_status_persistencia(contexto):
        """
        Define o status operacional inicial do chamado.

        O status operacional não altera a classificação
        prevista pelo modelo.
        """

        decisao = contexto.get("decision")

        if decisao == DECISAO_REJEITAR:
            return "REJEITADO"

        if decisao == DECISAO_ESCLARECER:
            return "AGUARDANDO_INFORMACOES"

        if decisao == DECISAO_ESCALONAR:
            return "BLOQUEADO_SEGURANCA"

        if decisao == "SERVICE_UNAVAILABLE":
            return "FALHA_TECNICA"

        if (
            contexto.get("needs_human_review")
            or contexto.get("classification_status")
            in {
                "SUGGESTED",
                "REVIEW_ONLY",
                "PROVISIONAL"
            }
        ):
            return "AGUARDANDO_REVISAO"

        return "NOVO"

    def renderizar_resposta(**campos):
        contexto_base = {
            "chamado": texto_chamado,
            "decision": None,
            "reason_code": None,
            "confidence_action": None,
            "confidence_level": None,
            "confidence_message": None,
            "confianca_categoria": None,
            "confianca_prioridade": None,
            "confianca_setor": None,
            "confianca_final": None,
            "triage_context": triage_context,
            "priority_predicted": None,
            "priority_source": "NONE",
            "classification_status": "REVIEW_ONLY",
            "classification_available": False,
            "classification_complete": False,
            "needs_human_review": False,
            "protocolo": None,
            "chamado_registrado": False,
            "message": None,
            "erro": None
        }

        contexto_base.update(campos)

        if texto_chamado:
            status_persistencia = (
                obter_status_persistencia(
                    contexto_base
                )
            )

            dados_chamado = {
                "descricao": texto_chamado,
                "decision": (
                    contexto_base.get("decision")
                    or "SERVICE_UNAVAILABLE"
                ),
                "reason_code": contexto_base.get(
                    "reason_code"
                ),
                "categoria": contexto_base.get(
                    "categoria"
                ),
                "prioridade": contexto_base.get(
                    "prioridade"
                ),
                "setor": contexto_base.get("setor"),
                "confianca_categoria": contexto_base.get(
                    "confianca_categoria"
                ),
                "confianca_prioridade": contexto_base.get(
                    "confianca_prioridade"
                ),
                "confianca_setor": contexto_base.get(
                    "confianca_setor"
                ),
                "confianca_final": contexto_base.get(
                    "confianca_final"
                ),
                "confidence_action": contexto_base.get(
                    "confidence_action"
                ),
                "confidence_level": contexto_base.get(
                    "confidence_level"
                ),
                "classification_status": contexto_base.get(
                    "classification_status"
                ),
                "classification_available": contexto_base.get(
                    "classification_available"
                ),
                "classification_complete": contexto_base.get(
                    "classification_complete"
                ),
                "needs_human_review": contexto_base.get(
                    "needs_human_review"
                ),
                "priority_predicted": contexto_base.get(
                    "priority_predicted"
                ),
                "priority_source": contexto_base.get(
                    "priority_source"
                ),
                "urgency_signal": triage_context.get(
                    "urgency_signal",
                    False
                ),
                "business_impact": triage_context.get(
                    "business_impact",
                    False
                ),
                "urgency_reason": triage_context.get(
                    "urgency_reason"
                ),
                "requested_urgency": triage_context.get(
                    "requested_urgency"
                ),
                "status": status_persistencia
            }

            resultado_persistencia = (
                registrar_chamado_no_banco(
                    dados_chamado
                )
            )

            contexto_base["protocolo"] = (
                resultado_persistencia["protocolo"]
            )

            contexto_base["chamado_registrado"] = True

        return render_template(
            "index.html",
            **contexto_base
        )

    if not texto_chamado:
        mensagem_campo_obrigatorio = (
            "Digite uma descrição para o chamado."
        )

        return renderizar_resposta(
            erro=mensagem_campo_obrigatorio,
            message=mensagem_campo_obrigatorio,
            decision=DECISAO_REJEITAR,
            reason_code="FIELD_MISSING"
        ), 400

    if len(texto_chamado) > TAMANHO_MAXIMO_CHAMADO:
        mensagem_tamanho = (
            "A descrição do chamado ultrapassa "
            "o limite permitido."
        )

        return renderizar_resposta(
            erro=mensagem_tamanho,
            message=mensagem_tamanho,
            decision=DECISAO_REJEITAR,
            reason_code="PAYLOAD_TOO_LARGE"
        ), 400

    # Etapa 14: executar guardrail antes dos modelos

    try:
        resultado_guardrail = avaliar_guardrail(
            texto_chamado
        )

    except Exception:
        mensagem_indisponibilidade = (
            "Não foi possível processar "
            "o chamado no momento."
        )

        return renderizar_resposta(
            erro=mensagem_indisponibilidade,
            message=mensagem_indisponibilidade,
            decision="SERVICE_UNAVAILABLE",
            reason_code="GUARDRAIL_UNAVAILABLE"
        ), 503

    decision_guardrail = resultado_guardrail.get(
        "decision"
    )

    reason_code_guardrail = resultado_guardrail.get(
        "reason_code"
    )

    mensagem_guardrail = resultado_guardrail.get(
        "message"
    )

    decisoes_permitidas = {
        DECISAO_ACEITAR,
        DECISAO_REJEITAR,
        DECISAO_ESCLARECER,
        DECISAO_ESCALONAR
    }

    if decision_guardrail not in decisoes_permitidas:
        mensagem_indisponibilidade = (
            "Não foi possível processar "
            "o chamado no momento."
        )

        return renderizar_resposta(
            erro=mensagem_indisponibilidade,
            message=mensagem_indisponibilidade,
            decision="SERVICE_UNAVAILABLE",
            reason_code="GUARDRAIL_UNAVAILABLE"
        ), 503

    if decision_guardrail == DECISAO_REJEITAR:
        return renderizar_resposta(
            erro=mensagem_guardrail,
            message=mensagem_guardrail,
            decision=decision_guardrail,
            reason_code=reason_code_guardrail
        ), 200

    if decision_guardrail == DECISAO_ESCLARECER:
        return renderizar_resposta(
            erro=mensagem_guardrail,
            message=mensagem_guardrail,
            decision=decision_guardrail,
            reason_code=reason_code_guardrail
        ), 200

    if decision_guardrail == DECISAO_ESCALONAR:
        return renderizar_resposta(
            erro=mensagem_guardrail,
            message=mensagem_guardrail,
            decision=decision_guardrail,
            reason_code=reason_code_guardrail
        ), 403

    # Etapa 15: executar modelos e obter probabilidades

    try:
        texto_categoria = vetorizador_categoria.transform(
            [texto_chamado]
        )

        categoria = modelo_categoria.predict(
            texto_categoria
        )[0]

        confianca_categoria = obter_confianca(
            modelo_categoria,
            texto_categoria,
            categoria
        )

        # Etapa 16: interromper a cadeia na categoria

        if confianca_categoria < LIMIAR_CONFIANCA_ESCLARECIMENTO:
            mensagem_revisao = (
                "O chamado foi recebido, mas precisa de revisão manual "
                "antes da classificação definitiva."
            )

            return renderizar_resposta(
                erro=mensagem_revisao,
                message=mensagem_revisao,
                decision=DECISAO_ACEITAR,
                confidence_action="LOW_CONFIDENCE_REVIEW",
                reason_code="LOW_CONFIDENCE_REVIEW",
                confidence_level="LOW_CONFIDENCE_REVIEW",
                classification_status="REVIEW_ONLY",
                classification_available=False,
                classification_complete=False,
                needs_human_review=True,
                priority_predicted=None,
                priority_source="NONE",
                confianca_categoria=confianca_categoria,
                confianca_final=confianca_categoria
            ), 200

        # Etapa 17: prever prioridade

        dados_prioridade = pd.DataFrame(
            [
                {
                    "texto": texto_chamado,
                    "categoria": categoria
                }
            ]
        )

        prioridade = modelo_prioridade.predict(
            dados_prioridade
        )[0]

        confianca_prioridade = obter_confianca(
            modelo_prioridade,
            dados_prioridade,
            prioridade
        )

        prioridade_baixa = (
            confianca_prioridade
            < LIMIAR_CONFIANCA_ESCLARECIMENTO
        )

        # A prioridade prevista continua sendo usada
        # como entrada do setor, mesmo com baixa confiança.

        # Etapa 18: prever setor

        dados_setor = pd.DataFrame(
            [
                {
                    "texto": texto_chamado,
                    "categoria": categoria,
                    "prioridade": prioridade
                }
            ]
        )

        setor = modelo_setor.predict(
            dados_setor
        )[0]

        confianca_setor = obter_confianca(
            modelo_setor,
            dados_setor,
            setor
        )

        # Confiança final conservadora

        confianca_final = min(
            confianca_categoria,
            confianca_prioridade,
            confianca_setor
        )

        resultado_confianca = classificar_nivel_confianca(
            confianca_final
        )

    except ErroConfiancaModelo:
        mensagem_confianca = (
            "Não foi possível validar a confiança "
            "da classificação no momento."
        )

        return renderizar_resposta(
            erro=mensagem_confianca,
            message=mensagem_confianca,
            decision="SERVICE_UNAVAILABLE",
            reason_code="MODEL_CONFIDENCE_UNAVAILABLE"
        ), 503

    except Exception:
        mensagem_modelo = (
            "Não foi possível realizar "
            "a classificação no momento."
        )

        return renderizar_resposta(
            erro=mensagem_modelo,
            message=mensagem_modelo,
            decision="SERVICE_UNAVAILABLE",
            reason_code="MODEL_UNAVAILABLE"
        ), 503

    # Etapa 19: tratar sugestão após os três modelos

    if (
        resultado_confianca["confidence_action"]
        == "LOW_CONFIDENCE_REVIEW"
    ):
        mensagem_revisao = (
            "A classificação abaixo é apenas uma sugestão "
            "e precisa de revisão humana antes da definição final."
        )

        return renderizar_resposta(
            chamado=texto_chamado,
            categoria=categoria,
            prioridade=prioridade,
            setor=setor,
            erro=mensagem_revisao,
            message=mensagem_revisao,
            decision=DECISAO_ACEITAR,
            confidence_action="LOW_CONFIDENCE_REVIEW",
            reason_code="LOW_CONFIDENCE_REVIEW",
            confidence_level="LOW_CONFIDENCE_REVIEW",
            confidence_message=mensagem_revisao,
            confianca_categoria=confianca_categoria,
            confianca_prioridade=confianca_prioridade,
            confianca_setor=confianca_setor,
            confianca_final=confianca_final,
            priority_predicted=prioridade,
            priority_source="MODEL_SUGGESTION",
            classification_status="SUGGESTED",
            classification_available=True,
            classification_complete=False,
            needs_human_review=True
        ), 200

    # Etapa 20: enviar resposta normal ou cautelosa para o HTML

    classification_status = (
        "DEFINITIVE"
        if resultado_confianca["confidence_action"] == "NORMAL"
        else "PROVISIONAL"
    )

    classification_complete = (
        classification_status == "DEFINITIVE"
    )

    needs_human_review = (
        resultado_confianca["confidence_action"] != "NORMAL"
    )

    return renderizar_resposta(
        chamado=texto_chamado,
        categoria=categoria,
        prioridade=prioridade,
        setor=setor,
        decision=resultado_confianca["decision"],
        confidence_action=resultado_confianca["confidence_action"],
        reason_code=reason_code_guardrail,
        confidence_level=resultado_confianca["nivel"],
        confidence_message=resultado_confianca["message"],
        confianca_categoria=confianca_categoria,
        confianca_prioridade=confianca_prioridade,
        confianca_setor=confianca_setor,
        confianca_final=confianca_final,
        priority_predicted=prioridade,
        priority_source="MODEL",
        classification_status=classification_status,
        classification_available=True,
        classification_complete=classification_complete,
        needs_human_review=needs_human_review,
        message=resultado_confianca["message"]
    )

# Etapa 20.1: exibir e processar cadastro de usuário pendente

@app.route("/cadastro", methods=["GET", "POST"])
def cadastro():
    if request.method == "GET":
        return render_template(
            "cadastro.html",
            csrf_token=obter_token_csrf(),
            cadastro_realizado=False,
            erro=None,
            usuario_email=""
        )

    validar_token_csrf()

    nome = request.form.get(
        "nome",
        ""
    ).strip()

    sobrenome = request.form.get(
        "sobrenome",
        ""
    ).strip()

    email = request.form.get(
        "email",
        ""
    ).strip().lower()

    senha = request.form.get(
        "senha",
        ""
    )

    try:
        validar_dados_cadastro(
            nome,
            sobrenome,
            email,
            senha
        )

        usuario = registrar_usuario_pendente(
            nome=nome,
            sobrenome=sobrenome,
            email=email,
            senha=senha
        )

    except ErroCadastro as erro:
        return render_template(
            "cadastro.html",
            csrf_token=obter_token_csrf(),
            cadastro_realizado=False,
            erro=str(erro),
            usuario_email=email
        ), 400

    except EmailJaCadastrado:
        return render_template(
            "cadastro.html",
            csrf_token=obter_token_csrf(),
            cadastro_realizado=False,
            erro=(
                "Já existe uma conta cadastrada "
                "com este e-mail."
            ),
            usuario_email=email
        ), 409

    except ErroBancoDados:
        return render_template(
            "cadastro.html",
            csrf_token=obter_token_csrf(),
            cadastro_realizado=False,
            erro=(
                "Não foi possível concluir o cadastro "
                "no momento."
            ),
            usuario_email=email
        ), 503

    return render_template(
        "cadastro.html",
        csrf_token=obter_token_csrf(),
        cadastro_realizado=True,
        erro=None,
        usuario_email=usuario["email"]
    ), 201


# Etapa 20.2: exibir e processar login

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template(
            "login.html",
            csrf_token=obter_token_csrf(),
            erro=None,
            usuario_email=""
        )

    validar_token_csrf()

    email = request.form.get(
        "email",
        ""
    ).strip().lower()

    senha = request.form.get(
        "senha",
        ""
    )

    if (
        not email
        or not senha
    ):
        return render_template(
            "login.html",
            csrf_token=obter_token_csrf(),
            erro="Informe e-mail e senha.",
            usuario_email=email
        ), 400

    try:
        usuario = obter_usuario_por_email(
            email
        )

    except ErroBancoDados:
        return render_template(
            "login.html",
            csrf_token=obter_token_csrf(),
            erro=(
                "Não foi possível validar o acesso "
                "no momento."
            ),
            usuario_email=email
        ), 503

    senha_valida = False

    if usuario is not None:
        try:
            senha_valida = check_password_hash(
                usuario["senha_hash"],
                senha
            )
        except ValueError:
            senha_valida = False

    if not senha_valida:
        return render_template(
            "login.html",
            csrf_token=obter_token_csrf(),
            erro="E-mail ou senha inválidos.",
            usuario_email=email
        ), 401

    if not bool(usuario["ativo"]):
        return render_template(
            "login.html",
            csrf_token=obter_token_csrf(),
            erro=(
                "Sua conta foi criada e aguarda "
                "liberação manual."
            ),
            usuario_email=email
        ), 403

    perfil = (
        usuario.get("perfil")
        or ""
    ).upper()

    if perfil not in PERFIS_AUTORIZADOS_DASHBOARD:
        return render_template(
            "login.html",
            csrf_token=obter_token_csrf(),
            erro="Sua conta não possui acesso ao dashboard.",
            usuario_email=email
        ), 403

    session.clear()

    session["usuario_admin_id"] = usuario["id"]
    session["usuario_admin_nome"] = usuario["nome"]
    session["dashboard_autorizado"] = True
    session["csrf_token"] = secrets.token_urlsafe(32)

    return redirect(
        url_for("dashboard")
    )


# Etapa 20.3: encerrar sessão autenticada

@app.route("/logout", methods=["POST"])
def logout():
    validar_token_csrf()

    session.clear()

    return redirect(
        url_for("login")
    )

# Etapa 21: preparar e carregar o dashboard com dados persistidos reais

ITENS_POR_PAGINA_DASHBOARD = 10

STATUS_FINALIZADOS_DASHBOARD = (
    "RESOLVIDO",
    "FECHADO",
    "CONCLUIDO",
    "CANCELADO"
)

ORDENACOES_DASHBOARD = {
    "recentes": "c.criado_em DESC, c.id DESC",
    "antigos": "c.criado_em ASC, c.id ASC",
    "protocolo": "c.protocolo ASC",
    "status": "c.status ASC, c.criado_em DESC, c.id DESC"
}

ROTULOS_STATUS_DASHBOARD = {
    "NOVO": "Aberto",
    "EM_ANDAMENTO": "Em atendimento",
    "AGUARDANDO_REVISAO": "Em revisão",
    "RESOLVIDO": "Concluído",
    "FECHADO": "Concluído",
    "CONCLUIDO": "Concluído",
    "CANCELADO": "Cancelado"
}


# Etapa 21.1: impedir exposição do dashboard sem sessão autorizada

def exigir_autorizacao_dashboard():
    """
    Impede o acesso público aos dados reais do dashboard.

    A futura rota de login deve preencher os dois valores
    somente após validar o usuário e o perfil autorizado.
    """

    usuario_admin_id = session.get(
        "usuario_admin_id"
    )

    dashboard_autorizado = session.get(
        "dashboard_autorizado"
    )

    if (
        not usuario_admin_id
        or dashboard_autorizado is not True
    ):
        abort(403)


# Etapa 21.2: validar datas recebidas pelos filtros

def validar_data_dashboard(valor_data):
    """
    Aceita somente datas no formato YYYY-MM-DD.

    Retorna string vazia quando o filtro não foi enviado.
    """

    if not valor_data:
        return ""

    try:
        datetime.strptime(
            valor_data,
            "%Y-%m-%d"
        )
    except ValueError:
        abort(400)

    return valor_data


# Etapa 21.3: definir rótulo seguro para o status do chamado

def obter_rotulo_status_dashboard(
    status,
    classification_status
):
    """
    REVIEW_ONLY não expõe classificação parcial.

    Mesmo que existam valores inconsistentes no banco,
    o estado REVIEW_ONLY permanece apresentado apenas
    como Em revisão.
    """

    if classification_status == "REVIEW_ONLY":
        return "Em revisão"

    return ROTULOS_STATUS_DASHBOARD.get(
        status,
        "Em atendimento"
    )


# Etapa 21.4: reduzir a descrição exibida na tabela

def resumir_descricao_dashboard(descricao):
    """
    Produz um resumo visual sem alterar a descrição salva.
    """

    texto_normalizado = " ".join(
        (descricao or "").split()
    )

    if not texto_normalizado:
        return "Sem descrição disponível."

    limite_resumo = 100

    if len(texto_normalizado) <= limite_resumo:
        return texto_normalizado

    return (
        texto_normalizado[:limite_resumo - 3].rstrip()
        + "..."
    )


# Etapa 21.5: obter opções reais para os filtros

def obter_opcoes_filtros_dashboard(cursor):
    """
    Lê somente valores já persistidos.

    As colunas são fixas e não recebem qualquer valor
    controlado pelo navegador.
    """

    configuracoes = {
        "categorias": {
            "coluna": "categoria",
            "rotulo": lambda valor: valor
        },
        "prioridades": {
            "coluna": "prioridade",
            "rotulo": lambda valor: valor
        },
        "setores": {
            "coluna": "setor",
            "rotulo": lambda valor: valor
        },
        "statuses": {
            "coluna": "status",
            "rotulo": lambda valor: (
                ROTULOS_STATUS_DASHBOARD.get(
                    valor,
                    "Em atendimento"
                )
            )
        }
    }

    opcoes = {}

    for nome_opcao, configuracao in configuracoes.items():
        coluna = configuracao["coluna"]

        query_opcoes = f"""
            SELECT DISTINCT
                {coluna} AS valor
            FROM chamados
            WHERE {coluna} IS NOT NULL
              AND {coluna} <> ''
            ORDER BY {coluna} ASC
        """

        cursor.execute(query_opcoes)

        registros = cursor.fetchall()

        opcoes[nome_opcao] = [
            {
                "value": registro["valor"],
                "label": configuracao["rotulo"](
                    registro["valor"]
                )
            }
            for registro in registros
        ]

    return opcoes


# Etapa 21.6: receber somente filtros permitidos

def obter_filtros_dashboard(opcoes):
    """
    Valida os filtros antes de montar a consulta.

    Valores desconhecidos retornam HTTP 400 e não chegam
    ao SQL como nome de coluna, ordenação ou fragmento SQL.
    """

    filtros = {
        "periodo_inicio": validar_data_dashboard(
            request.args.get(
                "periodo_inicio",
                ""
            ).strip()
        ),
        "periodo_fim": validar_data_dashboard(
            request.args.get(
                "periodo_fim",
                ""
            ).strip()
        ),
        "categoria": request.args.get(
            "categoria",
            ""
        ).strip(),
        "prioridade": request.args.get(
            "prioridade",
            ""
        ).strip(),
        "setor": request.args.get(
            "setor",
            ""
        ).strip(),
        "status": request.args.get(
            "status",
            ""
        ).strip(),
        "sort": request.args.get(
            "sort",
            "recentes"
        ).strip()
    }

    if (
        filtros["periodo_inicio"]
        and filtros["periodo_fim"]
        and filtros["periodo_inicio"]
        > filtros["periodo_fim"]
    ):
        abort(400)

    valores_permitidos = {
        "categoria": {
            item["value"]
            for item in opcoes["categorias"]
        },
        "prioridade": {
            item["value"]
            for item in opcoes["prioridades"]
        },
        "setor": {
            item["value"]
            for item in opcoes["setores"]
        },
        "status": {
            item["value"]
            for item in opcoes["statuses"]
        }
    }

    for nome_filtro, valores in valores_permitidos.items():
        if (
            filtros[nome_filtro]
            and filtros[nome_filtro] not in valores
        ):
            abort(400)

    if filtros["sort"] not in ORDENACOES_DASHBOARD:
        abort(400)

    return filtros


# Etapa 21.7: montar WHERE parametrizado para todas as consultas

def montar_filtros_sql_dashboard(filtros):
    """
    Retorna cláusulas SQL fixas e parâmetros separados.

    Nenhum valor recebido pela URL é concatenado ao SQL.
    """

    clausulas = []
    parametros = []

    if filtros["periodo_inicio"]:
        clausulas.append(
            "c.criado_em >= %s"
        )
        parametros.append(
            filtros["periodo_inicio"]
            + " 00:00:00"
        )

    if filtros["periodo_fim"]:
        clausulas.append(
            "c.criado_em < DATE_ADD(%s, INTERVAL 1 DAY)"
        )
        parametros.append(
            filtros["periodo_fim"]
        )

    filtros_colunas = {
        "categoria": "c.categoria",
        "prioridade": "c.prioridade",
        "setor": "c.setor",
        "status": "c.status"
    }

    for nome_filtro, coluna in filtros_colunas.items():
        if filtros[nome_filtro]:
            clausulas.append(
                f"{coluna} = %s"
            )
            parametros.append(
                filtros[nome_filtro]
            )

    if not clausulas:
        return "", parametros

    return (
        " WHERE "
        + " AND ".join(clausulas),
        parametros
    )


# Etapa 21.8: montar links que preservam filtros e ordenação

def montar_paginacao_dashboard(
    filtros,
    pagina,
    total_itens
):
    """
    Gera URLs de paginação preservando os filtros ativos.
    """

    total_paginas = max(
        1,
        (
            total_itens
            + ITENS_POR_PAGINA_DASHBOARD
            - 1
        )
        // ITENS_POR_PAGINA_DASHBOARD
    )

    argumentos_url = {
        nome: valor
        for nome, valor in filtros.items()
        if valor
    }

    urls_paginas = {}

    for numero_pagina in range(
        1,
        total_paginas + 1
    ):
        urls_paginas[numero_pagina] = url_for(
            "dashboard",
            page=numero_pagina,
            **argumentos_url
        )

    return {
        "page": pagina,
        "per_page": ITENS_POR_PAGINA_DASHBOARD,
        "total_items": total_itens,
        "total_pages": total_paginas,
        "previous_url": (
            urls_paginas.get(pagina - 1)
            if pagina > 1
            else None
        ),
        "next_url": (
            urls_paginas.get(pagina + 1)
            if pagina < total_paginas
            else None
        ),
        "page_urls": urls_paginas
    }


# Etapa 21.9: consultar os dados reais do dashboard

def obter_dados_dashboard():
    """
    Consulta exclusivamente registros persistidos em chamados.

    A rota não inventa números, cards ou gráficos.
    """

    conexao = None
    cursor = None

    try:
        conexao = obter_conexao_banco()
        cursor = conexao.cursor()

        opcoes = obter_opcoes_filtros_dashboard(
            cursor
        )

        filtros = obter_filtros_dashboard(
            opcoes
        )

        where_sql, parametros = (
            montar_filtros_sql_dashboard(
                filtros
            )
        )

        try:
            pagina = int(
                request.args.get(
                    "page",
                    "1"
                )
            )
        except ValueError:
            abort(400)

        if pagina < 1:
            abort(400)

        query_metricas = f"""
            SELECT
                COUNT(*) AS total,
                COALESCE(
                    SUM(
                        CASE
                            WHEN c.status NOT IN (
                                'RESOLVIDO',
                                'FECHADO',
                                'CONCLUIDO',
                                'CANCELADO'
                            )
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS open_count,
                COALESCE(
                    SUM(
                        CASE
                            WHEN c.classification_status = 'REVIEW_ONLY'
                              OR c.status = 'AGUARDANDO_REVISAO'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS review_count,
                COALESCE(
                    SUM(
                        CASE
                            WHEN c.status IN (
                                'RESOLVIDO',
                                'FECHADO',
                                'CONCLUIDO'
                            )
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS completed_count
            FROM chamados c
            {where_sql}
        """

        cursor.execute(
            query_metricas,
            tuple(parametros)
        )

        metricas_brutas = cursor.fetchone() or {}

        metrics = {
            "total": int(
                metricas_brutas.get("total", 0)
                or 0
            ),
            "open": int(
                metricas_brutas.get("open_count", 0)
                or 0
            ),
            "review": int(
                metricas_brutas.get("review_count", 0)
                or 0
            ),
            "completed": int(
                metricas_brutas.get(
                    "completed_count",
                    0
                )
                or 0
            )
        }

        clausula_categorias = (
            where_sql
            + (
                " AND "
                if where_sql
                else " WHERE "
            )
            + """
                c.status NOT IN (
                    'RESOLVIDO',
                    'FECHADO',
                    'CONCLUIDO',
                    'CANCELADO'
                )
                AND c.categoria IS NOT NULL
                AND c.categoria <> ''
                AND c.classification_status <> 'REVIEW_ONLY'
            """
        )

        query_categorias = f"""
            SELECT
                c.categoria AS name,
                COUNT(*) AS open_count
            FROM chamados c
            {clausula_categorias}
            GROUP BY c.categoria
            ORDER BY open_count DESC, name ASC
        """

        cursor.execute(
            query_categorias,
            tuple(parametros)
        )

        categories = [
            {
                "name": registro["name"],
                "open_count": int(
                    registro["open_count"]
                )
            }
            for registro in cursor.fetchall()
        ]

        configuracoes_graficos = {
            "priority": {
                "coluna": "c.prioridade",
                "condicao": """
                    c.prioridade IS NOT NULL
                    AND c.prioridade <> ''
                    AND c.classification_status <> 'REVIEW_ONLY'
                """
            },
            "sector": {
                "coluna": "c.setor",
                "condicao": """
                    c.setor IS NOT NULL
                    AND c.setor <> ''
                    AND c.classification_status <> 'REVIEW_ONLY'
                """
            },
            "status": {
                "coluna": """
                    CASE
                        WHEN c.classification_status = 'REVIEW_ONLY'
                        THEN 'Em revisão'
                        ELSE c.status
                    END
                """,
                "condicao": "1 = 1"
            },
            "daily": {
        "coluna": (
            "DATE_FORMAT(c.criado_em, '%%d/%%m')"
            ),
        "condicao": "1 = 1"
        }

        }

        charts = {}

        for nome_grafico, configuracao in (
            configuracoes_graficos.items()
        ):
            clausula_grafico = (
                where_sql
                + (
                    " AND "
                    if where_sql
                    else " WHERE "
                )
                + configuracao["condicao"]
            )

            query_grafico = f"""
                SELECT
                    {configuracao["coluna"]} AS label,
                    COUNT(*) AS value
                FROM chamados c
                {clausula_grafico}
                GROUP BY label
                ORDER BY value DESC, label ASC
            """

            cursor.execute(
                query_grafico,
                tuple(parametros)
            )

            charts[nome_grafico] = [
                {
                    "label": str(
                        registro["label"]
                    ),
                    "value": int(
                        registro["value"]
                    )
                }
                for registro in cursor.fetchall()
            ]

        query_total = f"""
            SELECT
                COUNT(*) AS total_items
            FROM chamados c
            {where_sql}
        """

        cursor.execute(
            query_total,
            tuple(parametros)
        )

        resultado_total = cursor.fetchone() or {}

        total_itens = int(
            resultado_total.get("total_items", 0)
            or 0
        )

        total_paginas = max(
            1,
            (
                total_itens
                + ITENS_POR_PAGINA_DASHBOARD
                - 1
            )
            // ITENS_POR_PAGINA_DASHBOARD
        )

        if pagina > total_paginas:
            pagina = total_paginas

        deslocamento = (
            pagina - 1
        ) * ITENS_POR_PAGINA_DASHBOARD

        ordenacao_sql = ORDENACOES_DASHBOARD[
            filtros["sort"]
        ]

        query_chamados = f"""
            SELECT
                c.protocolo,
                c.descricao,
                c.categoria,
                c.prioridade,
                c.setor,
                c.status,
                c.classification_status,
                c.criado_em
            FROM chamados c
            {where_sql}
            ORDER BY {ordenacao_sql}
            LIMIT %s OFFSET %s
        """

        parametros_chamados = (
            list(parametros)
            + [
                ITENS_POR_PAGINA_DASHBOARD,
                deslocamento
            ]
        )

        cursor.execute(
            query_chamados,
            tuple(parametros_chamados)
        )

        tickets = []

        for registro in cursor.fetchall():
            review_only = (
                registro.get("classification_status")
                == "REVIEW_ONLY"
            )

            criado_em = registro.get(
                "criado_em"
            )

            tickets.append(
                {
                    "protocol": registro.get(
                        "protocolo"
                    ),
                    "created_at": (
                        criado_em.isoformat()
                        if criado_em is not None
                        else None
                    ),
                    "created_at_label": (
                        criado_em.strftime(
                            "%d/%m/%Y %H:%M"
                        )
                        if criado_em is not None
                        else "—"
                    ),
                    "summary": (
                        resumir_descricao_dashboard(
                            registro.get("descricao")
                        )
                    ),
                    "category_label": (
                        "—"
                        if review_only
                        else (
                            registro.get("categoria")
                            or "—"
                        )
                    ),
                    "priority_label": (
                        "—"
                        if review_only
                        else (
                            registro.get("prioridade")
                            or "—"
                        )
                    ),
                    "sector_label": (
                        "—"
                        if review_only
                        else (
                            registro.get("setor")
                            or "—"
                        )
                    ),
                    "status": registro.get(
                        "status"
                    ),
                    "status_label": (
                        obter_rotulo_status_dashboard(
                            registro.get("status"),
                            registro.get(
                                "classification_status"
                            )
                        )
                    )
                }
            )

        pagination = montar_paginacao_dashboard(
            filtros,
            pagina,
            total_itens
        )

        return {
            "metrics": metrics,
            "categories": categories,
            "charts": charts,
            "tickets": tickets,
            "pagination": pagination,
            "filtros": filtros,
            "opcoes": opcoes
        }

    except ErroBancoDados:
        raise

    except pymysql.MySQLError as erro:
        raise ErroBancoDados(
            "Não foi possível consultar o dashboard."
        ) from erro

    finally:
        if cursor is not None:
            cursor.close()

        if conexao is not None:
            conexao.close()


# Etapa 21.10: renderizar o dashboard somente para sessão autorizada

@app.route("/dashboard", methods=["GET"])
def dashboard():
    exigir_autorizacao_dashboard()

    try:
        dados_dashboard = obter_dados_dashboard()

    except ErroBancoDados:
        return render_template(
            "dashboard.html",
            dashboard_state="error",
            dashboard_data=None,
            filtros={},
            opcoes={
                "categorias": [],
                "prioridades": [],
                "setores": [],
                "statuses": []
            }
        ), 503

    return render_template(
        "dashboard.html",
        dashboard_state="data",
        dashboard_data={
            "metrics": dados_dashboard["metrics"],
            "categories": dados_dashboard["categories"],
            "charts": dados_dashboard["charts"],
            "tickets": dados_dashboard["tickets"],
            "pagination": dados_dashboard["pagination"]
        },
        filtros=dados_dashboard["filtros"],
        opcoes=dados_dashboard["opcoes"]
    )

# Etapa 22: iniciar o servidor local

if __name__ == "__main__":
    app.run(debug=True)