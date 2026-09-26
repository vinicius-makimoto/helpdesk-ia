$(document).ready(function () {
  const $form = $("#formChamado");
  const $textarea = $("#chamado");
  const $contador = $("#contadorCaracteres");
  const $botao = $("#btnAnalisar");
  const $mensagemValidacao = $("#mensagemValidacao");

  function atualizarContador() {
    if (!$contador.length || !$textarea.length) {
      return;
    }

    $contador.text($textarea.val().length);
  }

  function abrirModalResultado() {
    const elementoModal = document.getElementById("modalResultado");

    if (
      !elementoModal ||
      typeof bootstrap === "undefined" ||
      !bootstrap.Modal
    ) {
      return;
    }

    bootstrap.Modal.getOrCreateInstance(elementoModal).show();
  }

  function obterDadosTriagem() {
    const elementoResultado = document.getElementById("resultadoTriagem");

    if (!elementoResultado) {
      return null;
    }

    return elementoResultado.dataset;
  }

  function valorBooleano(valor) {
    return (
      valor === true ||
      valor === "true" ||
      valor === "True" ||
      valor === "1" ||
      valor === 1
    );
  }

  function possuiImpactoFaturamento(valor) {
    const valorNormalizado = String(valor || "")
      .trim()
      .toUpperCase();

    return (
      valorNormalizado === "TRUE" ||
      valorNormalizado === "1" ||
      valorNormalizado === "BILLING_IMPACT" ||
      valorNormalizado === "REVENUE_IMPACT"
    );
  }

  function classificacaoDisponivel(dados) {
    return valorBooleano(dados.classificationAvailable);
  }

  function ehBaixaConfianca(dados) {
    const confidenceAction = String(dados.confidenceAction || "")
      .trim()
      .toUpperCase();

    const reasonCode = String(dados.reasonCode || "")
      .trim()
      .toUpperCase();

    const status = String(dados.status || "")
      .trim()
      .toUpperCase();

    return (
      confidenceAction === "LOW_CONFIDENCE_REVIEW" ||
      confidenceAction === "SUGGESTED" ||
      reasonCode === "LOW_CONFIDENCE_REVIEW" ||
      status === "SUGGESTED"
    );
  }

  function ehSuggested(dados) {
    return ehBaixaConfianca(dados) && classificacaoDisponivel(dados);
  }

  function ehReviewOnly(dados) {
    return ehBaixaConfianca(dados) && !classificacaoDisponivel(dados);
  }

  function exibirResultadoTriagem() {
    const dados = obterDadosTriagem();

    if (!dados || typeof Swal === "undefined") {
      return;
    }

    const decision = String(dados.decision || "")
      .trim()
      .toUpperCase();

    const reasonCode = String(dados.reasonCode || "")
      .trim()
      .toUpperCase();

    const confidenceAction = String(dados.confidenceAction || "")
      .trim()
      .toUpperCase();

    const status = String(dados.status || "")
      .trim()
      .toUpperCase();

    const businessImpact = String(dados.businessImpact || "").trim();

    const classificationAvailable = classificacaoDisponivel(dados);

    const hasBillingImpact = possuiImpactoFaturamento(businessImpact);

    const suggested = ehSuggested(dados);

    const reviewOnly = ehReviewOnly(dados);

    const message =
      dados.message || "Não foi possível obter detalhes sobre o resultado.";

    let alerta = null;
    let abrirModalDepois = false;

    /*
     * SUGGESTED:
     * há dados completos, mas eles são uma sugestão.
     */
    if (suggested) {
      const textoSugestao = hasBillingImpact
        ? "O chamado menciona possível impacto no faturamento. Essa informação será confirmada pela equipe de suporte."
        : "Esta é uma sugestão de classificação. A equipe poderá revisar estas informações.";

      alerta = {
        icon: "warning",
        title: hasBillingImpact
          ? "Revisão prioritária"
          : "Sugestão de classificação",
        text: textoSugestao,
        confirmButtonText: "Ver sugestão",
        confirmButtonColor: "#005de0",
      };

      abrirModalDepois = true;
    } else if (reviewOnly) {

    /*
     * REVIEW_ONLY:
     * não há classificação completa disponível.
     */
      const textoRevisao = hasBillingImpact
        ? "O chamado menciona possível impacto no faturamento. Essa informação será confirmada pela equipe de suporte."
        : "A equipe precisa analisar melhor o chamado antes de definir a classificação.";

      alerta = {
        icon: "warning",
        title: hasBillingImpact ? "Revisão prioritária" : "Análise preliminar",
        text: textoRevisao,
        confirmButtonText: "Entendi",
        confirmButtonColor: "#005de0",
      };
    } else if (confidenceAction === "CAUTION" && classificationAvailable) {

    /*
     * CAUTION com classificação disponível.
     */
      alerta = {
        icon: "warning",
        title: "Análise preliminar",
        text: "A classificação está disponível como uma análise preliminar. A equipe poderá revisar estas informações.",
        confirmButtonText: "Ver sugestão",
        confirmButtonColor: "#005de0",
      };

      abrirModalDepois = true;
    } else if (confidenceAction === "CAUTION" && !classificationAvailable) {

    /*
     * CAUTION sem classificação disponível.
     */
      alerta = {
        icon: "warning",
        title: "Análise preliminar",
        text: "A equipe precisa de mais informações antes de concluir a classificação.",
        confirmButtonText: "Entendi",
        confirmButtonColor: "#005de0",
      };
    } else if (

    /*
     * NORMAL:
     * abre a modal diretamente.
     */
      decision === "ACEITAR" &&
      confidenceAction === "NORMAL" &&
      classificationAvailable
    ) {
      abrirModalResultado();
      return;
    } else if (decision === "ESCLARECER") {
      alerta = {
        icon: "info",
        title: "Precisamos de mais detalhes",
        text: message,
        confirmButtonText: "Entendi",
        confirmButtonColor: "#005de0",
      };
    } else if (decision === "REJEITAR") {
      alerta = {
        icon: "warning",
        title: "Chamado não encaminhado",
        text: message,
        confirmButtonText: "Entendi",
        confirmButtonColor: "#005de0",
      };
    } else if (
      decision === "SECURITY_ESCALATION" ||
      decision === "SECURITY_ESCALATE"
    ) {
      alerta = {
        icon: "error",
        title: "Solicitação não processada",
        text: message,
        confirmButtonText: "Entendi",
        confirmButtonColor: "#005de0",
      };
    } else if (
      decision === "SERVICE_UNAVAILABLE" ||
      reasonCode === "SERVICE_UNAVAILABLE" ||
      status === "503"
    ) {
      alerta = {
        icon: "error",
        title: "Serviço temporariamente indisponível",
        text: message,
        confirmButtonText: "Tentar novamente",
        confirmButtonColor: "#005de0",
      };
    } else if (message) {
      alerta = {
        icon: "error",
        title: "Não foi possível analisar",
        text: message,
        confirmButtonText: "Entendi",
        confirmButtonColor: "#005de0",
      };
    }

    if (!alerta) {
      return;
    }

    Swal.fire(alerta).then(function () {
      if (abrirModalDepois) {
        abrirModalResultado();
      }

      if (decision === "ESCLARECER") {
        $textarea.trigger("focus");
      }
    });
  }

  atualizarContador();

  if ($textarea.length) {
    $textarea.on("input", function () {
      atualizarContador();

      $mensagemValidacao.addClass("d-none");
      $textarea.removeClass("is-invalid");
    });
  }

  if ($form.length) {
    $form.on("submit", function (event) {
      const descricao = String($textarea.val() || "").trim();

      if (descricao.length < 12) {
        event.preventDefault();

        $mensagemValidacao.removeClass("d-none");

        $textarea.addClass("is-invalid").trigger("focus");

        return;
      }

      $botao.addClass("is-loading").prop("disabled", true);
    });
  }

  exibirResultadoTriagem();
});
