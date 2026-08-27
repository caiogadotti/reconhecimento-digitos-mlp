<div align="center">

# Rede Neural para Dígitos Manuscritos

**A rede acerta 96% em dígitos escaneados. E em dígitos gerados por IA?**

Uma rede neural (MLP) treinada no dataset `digits` do scikit-learn, comparada
com o k-NN da Aula 2, e testada contra cinco dígitos manuscritos gerados por
IA para investigar se o modelo generaliza além do estilo do dataset de treino.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Licença](https://img.shields.io/badge/licen%C3%A7a-MIT-green)](LICENSE)

`Acurácia de 96,0% no conjunto de teste do dataset, contra 20,0% em dígitos escritos à mão gerados por IA`

**Português** &nbsp;·&nbsp; [English](README.en.md)

</div>

---

## O problema

O dataset `digits` do scikit-learn vem de formulários escaneados: traço fino,
uniforme, sem sombra. É um exemplo didático ótimo para treinar rápido, mas
levanta uma pergunta que o exercício guiado não responde sozinho: será que o
modelo aprendeu o **conceito** de cada dígito, ou só o estilo visual
específico daquele scan?

Este projeto responde isso testando o modelo em dígitos que ele nunca viu:
cinco imagens de dígitos escritos à mão, geradas por IA (modelo Z Image, via
Higgsfield), pedindo explicitamente um traço de caneta sobre papel branco,
foto realista - diferente da caligrafia binarizada do dataset original.

> Treinamos um modelo de **classificação multiclasse** que recebe **uma
> imagem 8×8 em tons de cinza (64 atributos)** e produz **o dígito de 0 a 9
> correspondente**.

| | |
|---|---|
| **Tipo de tarefa** | Classificação multiclasse (10 classes) |
| **Modelo** | `MLPClassifier`, camadas ocultas (64, 32), ativação ReLU |
| **Entrada `X`** | 64 valores de 0 a 16, a intensidade de cada pixel |
| **Saída `y`** | Um inteiro de 0 a 9 |
| **Métrica** | Acurácia em conjunto reservado + teste com imagens externas |

---

## Como rodar

```bash
git clone https://github.com/caiogadotti/reconhecimento-digitos-mlp.git
cd reconhecimento-digitos-mlp
pip install -r requirements.txt
streamlit run dashboard.py
```

O dashboard abre em `http://localhost:8501`, com duas abas: o exemplo guiado
(treino, curva de perda, matriz de confusão, comparação com k-NN) e o
desafio autoral (as cinco imagens, previsão de cada uma, e a investigação de
por que a acurácia cai).

O notebook original da aula (`aula04_rede_digitos.ipynb`) contém o mesmo
pipeline em formato de exercício, célula por célula, com as respostas do
desafio escritas no fechamento.

---

## Resultados do exemplo guiado

| Modelo | Acurácia no teste | Erros (de 450) |
|---|---:|---:|
| k-NN (k=3, Aula 2) | **98,4%** | 7 |
| Rede neural (MLP) | 96,0% | 18 |

O k-NN levou uma vantagem pequena. Não é incomum em datasets pequenos e
limpos como este (1.797 imagens, 8×8, praticamente sem ruído): com classes
bem separadas em 64 dimensões, olhar os vizinhos mais próximos já resolve
quase tudo, e a capacidade extra de uma rede neural não chega a compensar o
custo de treinar mais parâmetros com poucos dados.

---

## O desafio: por que a acurácia despenca pra 20%?

Rodando as cinco imagens geradas por IA pelo mesmo pipeline, o resultado foi
**1 acerto em 5 (20,0%)**, incluindo um erro com 99,7% de confiança. A
primeira hipótese foi o pré-processamento: a função que recorta e centraliza
o dígito usa um **limiar fixo** (`0.2 * valor máximo da imagem`) para separar
traço de fundo, e imagens foto-realistas têm sombra e textura que esse limiar
simples não isola bem.

Trocando pelo limiar de **Otsu** (adaptativo, calcula o ponto de corte ideal
para cada imagem), o recorte ficou visualmente correto - mas a acurácia
**não mudou**, continuou 1/5. Isso descartou a hipótese do recorte errado e
levou à real: comparando o histograma de intensidade de pixel do dataset de
treino contra o das imagens do desafio, a distribuição é visivelmente
diferente. No dataset de treino a maioria dos pixels está perto de 0 ou perto
do valor máximo (scan quase binarizado); nas imagens do desafio, a
distribuição é mais espalhada pelo meio da escala, efeito da sombra e do
antialiasing de uma foto real que sobrevive ao redimensionamento para 8×8.

| | Dataset de treino | Imagens do desafio |
|---|---:|---:|
| Média do pixel (0-16) | 4,88 | 4,37 |
| Desvio padrão | 6,02 | 4,52 |

A rede aprendeu pesos ajustados ao contraste quase binário do treino, e uma
imagem com distribuição intermediária cai fora da região do espaço de
entrada onde o modelo decide com confiança. O dashboard reproduz essa
comparação ao vivo, com o método de limiar selecionável.

---

## Estrutura do projeto

```
├── aula04_rede_digitos.ipynb   notebook da aula: exemplo guiado + desafio autoral
├── dashboard.py                dashboard Streamlit com as duas investigações
├── digitos_desafio/            as 5 imagens do desafio, geradas por IA
└── requirements.txt
```

---

## Stack

| Biblioteca | Papel |
|---|---|
| **scikit-learn** | `MLPClassifier`, `KNeighborsClassifier`, métricas, `train_test_split` |
| **scikit-image** | `threshold_otsu`, limiar adaptativo no pré-processamento |
| **NumPy / Pillow** | Recorte, centralização e redimensionamento das imagens externas |
| **Streamlit** | Dashboard interativo |
| **Matplotlib** | Curva de perda, matriz de confusão, histogramas |

---

## Créditos e licença

**Disciplina:** Laboratório Computacional de Aprendizado de Máquina (LCML), 2026/2
**Turma:** CIB-NA8
**Professor:** Reinaldo Augusto de Oliveira Ramos

**Dados:** `digits`: [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits),
E. Alpaydin e C. Kaynak, UCI Machine Learning Repository. Distribuído com o scikit-learn.

Código sob licença [MIT](LICENSE).
