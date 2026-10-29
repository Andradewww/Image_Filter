from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#FUNÇÃO AUXILIAR
def reconstruir_imagem(frequencias):
    """Converte as frequências novamente em uma imagem."""
    frequencias_na_ordem_original = np.fft.ifftshift(frequencias)
    imagem_reconstruida = np.fft.ifft2(frequencias_na_ordem_original)

    return imagem_reconstruida.real



#1.ABRE A IMAGEM EM TONS DE CINZA

pasta_projeto = Path(__file__).resolve().parent
caminho_imagem = pasta_projeto / "imagens" / "ImgEntrada.jpeg"

foto = Image.open(caminho_imagem)
foto_cinza = foto.convert("L")

imagem_original = np.array(foto_cinza, dtype=float)


#2.APLICA A TRANSFORMADA DE FOURIER

#transforma os valores dos pixels em componentes de frequencia
frequencias = np.fft.fft2(imagem_original)

#organiza as baixas frequências no centro.
frequencias_centralizadas = np.fft.fftshift(frequencias)

#prepara o espectro apenas para visualização
espectro = np.log1p(np.abs(frequencias_centralizadas))


#3.RECONSTROI A IMAGEM SEM FILTRAGEM

imagem_reconstruida = reconstruir_imagem(frequencias_centralizadas)


#4.CRIA OS FILTROS

altura, largura = imagem_original.shape

#calcula as frequencias de cada direcao em ciclos por pixel
frequencias_verticais = np.fft.fftfreq(altura)
frequencias_horizontais = np.fft.fftfreq(largura)

#centraliza as frequencias e monta uma grade bidimensional
frequencias_verticais = np.fft.fftshift(frequencias_verticais)
frequencias_horizontais = np.fft.fftshift(frequencias_horizontais)

grade_horizontal, grade_vertical = np.meshgrid(
    frequencias_horizontais,
    frequencias_verticais
)

distancia = np.sqrt(grade_horizontal**2 + grade_vertical**2)

#valores teste 0.01, 0.025 ou 0.08
#no passa-baixa, valores menores produzem mais suavização
largura_filtro = 0.025

#mascara gaussiana: preserva o centro e atenua as altas frequências
mascara_baixa = np.exp(
    -(distancia**2) / (2 * largura_filtro**2)
)

#a mascara complementar atenua o centro e preserva altas frequências
mascara_alta = 1 - mascara_baixa


#5.APLICA O PASSA-BAIXA

frequencias_baixa = frequencias_centralizadas * mascara_baixa
imagem_passa_baixa = reconstruir_imagem(frequencias_baixa)


#6.APLICAR O PASSA-ALTA

frequencias_alta = frequencias_centralizadas * mascara_alta
imagem_passa_alta = reconstruir_imagem(frequencias_alta)


#ajuste de exibicao: zero aparece como cinza
limite = max(float(np.max(np.abs(imagem_passa_alta))), 1.0)


# 7. SALVAR AS IMAGENS

pasta_saida = pasta_projeto / "imagens_saida"
pasta_saida.mkdir(exist_ok=True)

#usa as mesmas escalas de intensidade da exibicao
plt.imsave(
    pasta_saida / "01_original_cinza.png",
    imagem_original,
    cmap="gray", vmin=0, vmax=255
)

plt.imsave(
    pasta_saida / "02_espectro_fourier.png",
    espectro,
    cmap="gray"
)

plt.imsave(
    pasta_saida / "03_reconstrucao_sem_filtro.png",
    imagem_reconstruida,
    cmap="gray", vmin=0, vmax=255
)

plt.imsave(
    pasta_saida / "04_mascara_passa_baixa.png",
    mascara_baixa,
    cmap="gray", vmin=0, vmax=1
)

plt.imsave(
    pasta_saida / "05_resultado_passa_baixa.png",
    imagem_passa_baixa,
    cmap="gray", vmin=0, vmax=255
)

plt.imsave(
    pasta_saida / "06_resultado_passa_alta.png",
    imagem_passa_alta,
    cmap="gray", vmin=-limite, vmax=limite
)

print(f"Imagens salvas em: {pasta_saida}")
