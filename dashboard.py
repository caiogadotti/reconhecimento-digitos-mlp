from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion
from skimage.filters import threshold_otsu
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
PASTA_DESAFIO = BASE_DIR / "digitos_desafio"

st.set_page_config(page_title="Aula 4 - Redes Neurais", page_icon="🔢", layout="wide")


@st.cache_resource
def treinar_modelos():
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    rede = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
        )),
    ])
    rede.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred_rede = rede.predict(X_test)
    y_pred_knn = knn.predict(X_test)

    return {
        "digits": digits,
        "X": X,
        "rede": rede,
        "knn": knn,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_rede": y_pred_rede,
        "y_pred_knn": y_pred_knn,
        "acc_rede": accuracy_score(y_test, y_pred_rede),
        "acc_knn": accuracy_score(y_test, y_pred_knn),
        "epocas": rede.named_steps["mlp"].n_iter_,
        "loss_curve": rede.named_steps["mlp"].loss_curve_,
    }


def augmentar_digito(imagem_8x8, rng):
    """Varia espessura (dilatacao/erosao), aplica desfoque leve e ruido, simulando foto real."""
    im = imagem_8x8.copy()
    operacao = rng.choice(["dilatar", "erodir", "nenhuma"])
    if operacao == "dilatar":
        im = grey_dilation(im, size=(2, 2))
    elif operacao == "erodir":
        im = grey_erosion(im, size=(2, 2))
    if rng.random() < 0.7:
        im = gaussian_filter(im, sigma=rng.uniform(0.3, 0.9))
    if rng.random() < 0.6:
        im = im + rng.normal(0, rng.uniform(0.3, 1.2), im.shape)
    return np.clip(im, 0, 16)


@st.cache_resource
def treinar_modelo_aumentado(_X_train, _y_train, _X_test, _y_test, n_copias=3):
    rng = np.random.default_rng(42)
    X_partes = [_X_train]
    y_partes = [_y_train]
    for _ in range(n_copias):
        copias = np.array([augmentar_digito(img.reshape(8, 8), rng).ravel() for img in _X_train])
        X_partes.append(copias)
        y_partes.append(_y_train)
    X_aumentado = np.vstack(X_partes)
    y_aumentado = np.concatenate(y_partes)

    rede = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
        )),
    ])
    rede.fit(X_aumentado, y_aumentado)
    return {
        "rede": rede,
        "acc_teste": accuracy_score(_y_test, rede.predict(_X_test)),
        "n_treino": X_aumentado.shape[0],
    }


def preprocess_external_digit(path, threshold_frac=0.2, metodo="fixo"):
    """Recorta o digito, centraliza e reduz pra 8x8 em escala 0-16 (mesmo formato do dataset digits)."""
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=float)
    if array.mean() > 127:
        array = 255 - array
    if metodo == "otsu":
        try:
            threshold = threshold_otsu(array)
        except ValueError:
            threshold = max(30, threshold_frac * array.max())
    else:
        threshold = max(30, threshold_frac * array.max())
    coordinates = np.argwhere(array > threshold)
    if coordinates.size == 0:
        return None
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    crop = array[y_min:y_max, x_min:x_max]
    side = max(crop.shape)
    margin = max(2, side // 5)
    canvas = np.zeros((side + 2 * margin, side + 2 * margin))
    top = (canvas.shape[0] - crop.shape[0]) // 2
    left = (canvas.shape[1] - crop.shape[1]) // 2
    canvas[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    small = Image.fromarray(canvas.astype(np.uint8)).resize((8, 8), Image.Resampling.LANCZOS)
    result = np.asarray(small, dtype=float)
    if result.max() > 0:
        result = 16 * result / result.max()
    return result


@st.cache_data
def avaliar_desafio(_rede, threshold_frac=0.2, metodo="fixo", modelo_tag="base"):
    # modelo_tag existe só pra entrar na chave do cache (o parâmetro _rede, com
    # underscore, é ignorado pelo st.cache_data) e diferenciar rede original de aumentada.
    arquivos = sorted(PASTA_DESAFIO.glob("*.png")) + sorted(PASTA_DESAFIO.glob("*.jpg"))
    linhas = []
    for path in arquivos:
        img = preprocess_external_digit(path, threshold_frac, metodo)
        if img is None:
            continue
        real = int(path.stem[0])
        vetor = img.reshape(1, -1)
        previsto = int(_rede.predict(vetor)[0])
        probabilidades = _rede.predict_proba(vetor)[0]
        linhas.append({
            "arquivo": path.name,
            "real": real,
            "previsto": previsto,
            "correto": real == previsto,
            "confianca": float(probabilidades.max()),
            "imagem_8x8": img,
        })
    return linhas


st.title("Aula 4 - reconhecimento de dígitos com rede neural")
st.markdown(
    "Este dashboard reexecuta o pipeline do notebook `aula04_rede_digitos.ipynb`: "
    "treina um `MLPClassifier` no dataset `digits` do scikit-learn e testa o modelo "
    "em cinco fotos de dígitos escritos à mão, geradas por IA como parte do desafio "
    "autoral da aula."
)

dados = treinar_modelos()

aba_guiado, aba_desafio = st.tabs(["Exemplo guiado", "Desafio autoral"])

with aba_guiado:
    st.header("Como a rede foi montada")
    st.markdown(
        "Cada imagem do dataset `digits` é uma matriz 8×8 de pixels em tons de cinza "
        "(0 a 16), achatada em um vetor de **64 números**. A rede usada é um "
        "`MLPClassifier(hidden_layer_sizes=(64, 32))`: uma camada de entrada com 64 "
        "neurônios (um por pixel), uma camada oculta de 64 neurônios, outra de 32, e "
        "uma camada de saída com 10 neurônios (um por dígito, 0 a 9), treinada por "
        "retropropagação de erro (`backpropagation`)."
    )
    st.markdown(
        "Antes de entrar na rede, os pixels passam por `StandardScaler` - sem isso, "
        "pixels com variância alta (bordas do dígito, que variam muito de imagem pra "
        "imagem) dominariam o treinamento sobre pixels de variância baixa (cantos "
        "sempre pretos), atrasando a convergência."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia - rede neural", f"{dados['acc_rede']:.1%}")
    col1.caption("Proporção de acertos nas 450 imagens de teste, nunca vistas no treino.")
    col2.metric("Acurácia - k-NN (k=3)", f"{dados['acc_knn']:.1%}")
    col2.caption("Mesma divisão treino/teste, pra comparação justa com a Aula 2.")
    col3.metric("Épocas até parar", dados["epocas"])
    col3.caption("`early_stopping=True` interrompe o treino quando o erro de validação para de cair.")

    st.subheader("Curva de perda do treinamento")
    st.markdown(
        "A perda (`loss`) mede o quão erradas estão as previsões da rede a cada época. "
        "Ela deve cair de forma consistente; se ela oscilar ou subir, é sinal de taxa "
        "de aprendizado alta demais ou de instabilidade no treino."
    )
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(dados["loss_curve"], color="#167D7F", linewidth=2)
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda (log-loss)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Matriz de confusão")
        st.markdown(
            "Linha = classe real, coluna = classe prevista. A diagonal são os acertos; "
            "qualquer valor fora da diagonal é uma confusão específica entre dois dígitos."
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(
            dados["y_test"], dados["y_pred_rede"],
            display_labels=dados["digits"].target_names,
            cmap="Purples", ax=ax, colorbar=False,
        )
        st.pyplot(fig)
    with col_b:
        st.subheader("Rede neural vs k-NN")
        comparacao = pd.DataFrame({
            "modelo": ["k-NN (k=3)", "Rede neural (MLP)"],
            "acurácia (%)": [dados["acc_knn"] * 100, dados["acc_rede"] * 100],
            "erros (de 450)": [
                int((dados["y_test"] != dados["y_pred_knn"]).sum()),
                int((dados["y_test"] != dados["y_pred_rede"]).sum()),
            ],
        })
        st.dataframe(comparacao, hide_index=True, use_container_width=True)
        st.markdown(
            "O k-NN venceu por uma margem pequena (98,4% contra 96,0%). Isso não é "
            "incomum em datasets pequenos e limpos como este (1.797 imagens, 8×8, sem "
            "ruído): com poucas classes bem separadas no espaço de 64 dimensões, olhar "
            "os vizinhos mais próximos já resolve quase tudo, e a capacidade extra de "
            "uma rede neural (aprender combinações não-lineares de pixels) não chega a "
            "compensar o custo de treinar mais parâmetros com poucos dados."
        )

    st.subheader("Onde a rede errou no teste")
    erros_idx = np.flatnonzero(dados["y_test"] != dados["y_pred_rede"])
    if len(erros_idx):
        st.caption(f"{len(erros_idx)} erros em 450 imagens de teste. Mostrando até 10.")
        cols = st.columns(5)
        for i, idx in enumerate(erros_idx[:10]):
            with cols[i % 5]:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(dados["X_test"][idx].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
                ax.set_title(f"real {dados['y_test'][idx]} · previu {dados['y_pred_rede'][idx]}", fontsize=8)
                ax.axis("off")
                st.pyplot(fig)
    else:
        st.info("A rede não errou nenhuma amostra do conjunto de teste.")

with aba_desafio:
    st.header("O desafio: a rede reconhece dígitos que ela nunca viu?")
    st.markdown(
        "O dataset `digits` vem de formulários escaneados: traço fino, uniforme, sem "
        "sombra. As cinco imagens abaixo não são desse jeito - foram geradas pelo "
        "modelo Z Image (Higgsfield), pedindo um dígito escrito à mão com caneta preta "
        "sobre papel branco, em estilo de foto real. É um teste de generalização: será "
        "que o modelo aprendeu o *conceito* de cada dígito, ou só o estilo específico "
        "do dataset de treino?"
    )

    col_metodo, col_modelo = st.columns(2)
    with col_metodo:
        metodo = st.radio(
            "Como separar o traço do fundo antes de reduzir a imagem para 8×8?",
            options=["fixo", "otsu"],
            format_func=lambda m: "Limiar fixo (o que o notebook usa)" if m == "fixo" else "Limiar de Otsu (adaptativo)",
        )
        st.caption(
            "Limiar fixo: qualquer pixel acima de 20% do valor máximo da imagem "
            "(`0.2 * array.max()`) é considerado \"traço\". Quebra se a imagem tiver "
            "sombra ou textura de fundo clara. Otsu: calcula automaticamente o ponto "
            "de corte que melhor separa traço de fundo, minimizando a variância "
            "dentro de cada grupo."
        )
    with col_modelo:
        usar_aumentada = st.radio(
            "Qual rede usar?",
            options=[False, True],
            format_func=lambda v: "Rede original (só dataset digits)" if not v else "Rede com data augmentation",
        )
        st.caption(
            "Data augmentation: para cada imagem de treino, gera cópias com variação "
            "de espessura do traço, desfoque leve e ruído, simulando o estilo de uma "
            "foto real, e treina incluindo essas cópias junto com os dados originais."
        )

    threshold_frac = st.slider(
        "Fração do pico usada como limiar (só entra em ação no método \"fixo\")",
        min_value=0.05, max_value=0.6, value=0.2, step=0.05,
        disabled=metodo == "otsu",
    )

    if usar_aumentada:
        modelo_aumentado = treinar_modelo_aumentado(dados["X_train"], dados["y_train"], dados["X_test"], dados["y_test"])
        rede_usada = modelo_aumentado["rede"]
        st.caption(f"Rede treinada com {modelo_aumentado['n_treino']} imagens (dataset original + 3 cópias aumentadas). Acurácia no teste padrão: {modelo_aumentado['acc_teste']:.1%}.")
    else:
        rede_usada = dados["rede"]

    resultados = avaliar_desafio(rede_usada, threshold_frac, metodo, "aumentada" if usar_aumentada else "base")

    if not resultados:
        st.warning(f"Nenhuma imagem encontrada em {PASTA_DESAFIO}.")
    else:
        acertos = sum(r["correto"] for r in resultados)
        acuracia_desafio = acertos / len(resultados)
        st.metric(
            "Acurácia nas imagens do grupo",
            f"{acuracia_desafio:.1%}",
            f"{acertos}/{len(resultados)} corretas",
            delta_color="off",
        )

        cols = st.columns(len(resultados))
        for col, r in zip(cols, resultados):
            with col:
                st.image(str(PASTA_DESAFIO / r["arquivo"]), use_container_width=True)
                veredito = "acertou" if r["correto"] else "errou"
                st.markdown(f"**real {r['real']} · previsto {r['previsto']}** - {veredito}")
                st.caption(f"confiança da rede: {r['confianca']:.1%}")
                fig, ax = plt.subplots(figsize=(1.6, 1.6))
                ax.imshow(r["imagem_8x8"], cmap="gray_r", vmin=0, vmax=16)
                ax.axis("off")
                ax.set_title("o que a rede recebe", fontsize=7)
                st.pyplot(fig)

        st.divider()
        st.subheader("Por que a acurácia é tão mais baixa aqui que no teste?")

        media_treino = dados["X"].mean()
        desvio_treino = dados["X"].std()
        pixels_desafio = np.concatenate([r["imagem_8x8"].ravel() for r in resultados])
        media_desafio = pixels_desafio.mean()
        desvio_desafio = pixels_desafio.std()

        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("**Distribuição dos valores de pixel - dataset de treino**")
            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.hist(dados["X"].ravel(), bins=17, range=(0, 16), color="#167D7F")
            ax.set_xlabel("valor do pixel (0-16)")
            ax.set_ylabel("contagem")
            st.pyplot(fig)
            st.caption(f"média {media_treino:.2f} · desvio padrão {desvio_treino:.2f}")
        with col_y:
            st.markdown("**Distribuição dos valores de pixel - imagens do desafio**")
            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.hist(pixels_desafio, bins=17, range=(0, 16), color="#8a4fff")
            ax.set_xlabel("valor do pixel (0-16)")
            ax.set_ylabel("contagem")
            st.pyplot(fig)
            st.caption(f"média {media_desafio:.2f} · desvio padrão {desvio_desafio:.2f}")

        st.markdown(
            "Os dois histogramas não têm a mesma forma. No dataset de treino a maioria "
            "dos pixels está perto de 0 (fundo) com um grupo separado perto do valor "
            "máximo (o traço), porque o scan original já vem quase binarizado. Nas "
            "imagens do desafio a distribuição é mais espalhada pelo meio da escala - "
            "efeito da sombra suave e do antialiasing de uma foto real, que o "
            "redimensionamento pra 8×8 borra ainda mais. A rede aprendeu pesos "
            "ajustados ao contraste \"quase binário\" do treino, então uma imagem com "
            "essa distribuição intermediária cai fora da região do espaço de entrada "
            "onde ela decide com confiança."
        )
        st.markdown(
            "Testando as quatro combinações de método de limiar (fixo/Otsu) e rede "
            "(original/com data augmentation), só uma chega a 2/5 (40%): **rede "
            "aumentada + limiar de Otsu**. Nenhuma melhoria isolada resolve sozinha - "
            "usar só Otsu com a rede original continua em 1/5, e treinar com "
            "augmentation mas manter o limiar fixo também continua em 1/5. As duas "
            "causas (recorte impreciso e estilo de traço diferente do treino) são "
            "independentes: corrigir só uma não é suficiente, mas mesmo corrigindo as "
            "duas juntas o resultado (40%) fica longe dos 96% do dataset original. "
            "Isso sugere que data augmentation sintético tem um limite: o próximo "
            "passo de verdade seria treinar com fotos reais de dígitos escritos à "
            "mão, com a textura genuína de traço de caneta em vez de uma aproximação."
        )
