document.addEventListener("DOMContentLoaded", function () {
  const formFiltros = document.querySelector("#formFiltrosDashboard");
  const btnAtualizar = document.querySelector("#btnAtualizarDashboard");
  const btnTentarNovamente = document.querySelector("#btnTentarNovamente");

  configurarFormularioFiltros(formFiltros);
  configurarBotaoAtualizar(btnAtualizar);
  configurarBotaoTentarNovamente(btnTentarNovamente);
  renderizarGraficos();
});

function configurarFormularioFiltros(formFiltros) {
  if (!formFiltros) {
    return;
  }

  formFiltros.addEventListener("submit", function () {
    const botaoEnviar = formFiltros.querySelector('button[type="submit"]');

    if (!botaoEnviar) {
      return;
    }

    botaoEnviar.disabled = true;
    botaoEnviar.innerHTML = `
      <span
        class="spinner-border spinner-border-sm me-2"
        aria-hidden="true"
      ></span>
      Aplicando filtros...
    `;
  });
}

function configurarBotaoAtualizar(btnAtualizar) {
  if (!btnAtualizar) {
    return;
  }

  btnAtualizar.addEventListener("click", function () {
    btnAtualizar.disabled = true;
    btnAtualizar.innerHTML = `
      <span
        class="spinner-border spinner-border-sm me-2"
        aria-hidden="true"
      ></span>
      Atualizando...
    `;

    window.location.reload();
  });
}

function configurarBotaoTentarNovamente(btnTentarNovamente) {
  if (!btnTentarNovamente) {
    return;
  }

  btnTentarNovamente.addEventListener("click", function () {
    btnTentarNovamente.disabled = true;
    btnTentarNovamente.innerHTML = `
      <span
        class="spinner-border spinner-border-sm me-2"
        aria-hidden="true"
      ></span>
      Tentando novamente...
    `;

    window.location.reload();
  });
}

function obterDadosGraficos() {
  const elementoDados = document.querySelector("#dashboardChartData");

  if (!elementoDados) {
    return {};
  }

  try {
    return JSON.parse(elementoDados.textContent);
  } catch (erro) {
    console.warn("Não foi possível interpretar os dados dos gráficos.", erro);
    return {};
  }
}

function renderizarGraficos() {
  const dadosGraficos = obterDadosGraficos();
  const containers = document.querySelectorAll(".dashboard-chart");

  containers.forEach(function (container) {
    const chave = container.dataset.chartKey;
    const itens = Array.isArray(dadosGraficos[chave])
      ? dadosGraficos[chave]
      : [];

    renderizarGrafico(container, itens);
  });
}

function renderizarGrafico(container, itens) {
  container.innerHTML = "";

  if (!itens.length) {
    const estadoVazio = document.createElement("div");

    estadoVazio.className = "dashboard-chart-empty";
    estadoVazio.textContent = "Nenhum dado disponível ainda.";

    container.appendChild(estadoVazio);
    return;
  }

  const maiorValor = Math.max(
    ...itens.map(function (item) {
      return Number(item.value) || 0;
    }),
  );

  itens.forEach(function (item) {
    const rotulo = item.label || "Não informado";
    const valor = Number(item.value) || 0;
    const percentual = maiorValor > 0 ? (valor / maiorValor) * 100 : 0;

    const linha = document.createElement("div");
    linha.className = "dashboard-chart-row";

    const label = document.createElement("span");
    label.className = "dashboard-chart-label";
    label.textContent = rotulo;
    label.title = rotulo;

    const trilha = document.createElement("div");
    trilha.className = "dashboard-chart-track";

    const barra = document.createElement("div");
    barra.className = "dashboard-chart-bar";
    barra.style.width = `${percentual}%`;

    const numero = document.createElement("strong");
    numero.className = "dashboard-chart-value";
    numero.textContent = valor;

    trilha.appendChild(barra);
    linha.appendChild(label);
    linha.appendChild(trilha);
    linha.appendChild(numero);
    container.appendChild(linha);
  });
}
