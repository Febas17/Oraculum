import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from matplotlib.colors import LogNorm, Normalize

def ajustar_contraste_canal(canal, limiares=(2,98)):
    """
    Uma imagem de satélite, ao invés de guardar as cores de cada pixel em 8 bits (r,g,b) (0-255) com uma faixa dinâmica baixa, guarda os dados com 16 bits (0-65535),
    uma faixa dinâmica alta (HDR), se tentarmos exibir a imagem primitiva, como as telas de notebook e celular geralmente são configuradas para 0-255, seria exibida
    uma imagem preta com baixíssimo contraste, por isso, essa função pega a faixa enorme e converte de forma inteligente para 0-255, realçando os detalhes visíveis
    ignorando valores extremos (pixels muito escuros ou reflexos muito brilhantes), que são como "ruído", e foca somente na faixa de dados com a informação relevante
    para a análise do terreno.
    """
    p_min, p_max = np.percentile(canal,limiares) #Guarda o "limiar" dos x% mais escuros e y% mais claros dos pixels (2 e 98% por padrão) (basicamente isola os intervalos de possíveis outliers
    canal_cortado = np.clip(canal, p_min, p_max) #"Limpa" a imagem, fazendo com que qualquer pixel mais escuro que p_min = p_min e qualquer pixel mais claro que p_max = p_max ("normaliza" os outliers)
    canal_ajustado = cv2.normalize(canal_cortado, None, 0, 255, cv2.NORM_MINMAX) #(p_min,p_max) = (0,255), todos os outros valores são distribuídos nesse intervalo, fazendo com que toda a faixa de dados fique visível
    canal_final = canal_ajustado.astype(np.uint8) #converte o tipo de dados da matriz já ajustada para dados inteiros de 8 bits, agora sim com o intervalo 0-255 esperado por bibliotecas como matplotlib
    return canal_final


def alinhar_imagens(caminho_satelite, caminho_relevo):
    """
    Carrega, reprojeta e alinha as imagens. Retorna 5 produtos:
    1. Imagem RGB (para visualização)
    2. Mapa de Relevo (para análise)
    3. Banda Verde Bruta (para cálculos)
    4. Banda NIR Bruta (para cálculos)
    5. Valor Nodata do relevo
    """
    print("Iniciando carregamento e alinhamento (versão final)...")

    # 1. Define a grade de destino RETA
    with rasterio.open(caminho_satelite) as src_sat:
        bounds, res = src_sat.bounds, src_sat.res[0]
        dst_width = int(np.ceil((bounds.right - bounds.left) / res))
        dst_height = int(np.ceil((bounds.top - bounds.bottom) / res))
        dst_transform = rasterio.transform.from_origin(bounds.left, bounds.top, res, res)
        dst_profile = src_sat.profile.copy()
        dst_profile.update({'transform': dst_transform, 'width': dst_width, 'height': dst_height, 'nodata': 0})

    # 2. Cria os "moldes"
    satelite_alinhado_raw = np.zeros((dst_profile['count'], dst_height, dst_width), dtype=dst_profile['dtype'])
    with rasterio.open(caminho_relevo) as src_rel:
        relevo_alinhado_raw = np.zeros((1, dst_height, dst_width), dtype=src_rel.profile['dtype'])
        nodata_value_relevo = src_rel.nodata

    # 3. Reprojeta AMBAS as imagens
    with rasterio.open(caminho_satelite) as src:
        reproject(
            source=rasterio.band(src, src.indexes), destination=satelite_alinhado_raw,
            dst_transform=dst_transform, dst_crs=dst_profile['crs'], resampling=Resampling.bilinear
        )
    with rasterio.open(caminho_relevo) as src:
        reproject(
            source=rasterio.band(src, 1), destination=relevo_alinhado_raw,
            dst_transform=dst_transform, dst_crs=dst_profile['crs'], resampling=Resampling.bilinear,
            dst_nodata=nodata_value_relevo
        )
    print("Reprojeção concluída!")

    # 4. Separa os produtos
    imagem_4_bandas_cv = np.transpose(satelite_alinhado_raw, (1, 2, 0))
    # Ordem das bandas no arquivo do GEE: B4,B3,B2,B8 -> R, G, B, NIR
    r_bruto, g_bruto, b_bruto, nir_bruto = cv2.split(imagem_4_bandas_cv)

    # Cria a versão VISUAL RGB
    r_ajustado = ajustar_contraste_canal(r_bruto)
    g_ajustado = ajustar_contraste_canal(g_bruto)
    b_ajustado = ajustar_contraste_canal(b_bruto)
    imagem_satelite_rgb_final = cv2.merge([r_ajustado, g_ajustado, b_ajustado])

    imagem_relevo_final = relevo_alinhado_raw[0]

    print("Processamento finalizado.")
    # Retorna os produtos separados: um para ver, os outros para calcular
    return imagem_satelite_rgb_final, imagem_relevo_final, r_bruto, g_bruto, nir_bruto, nodata_value_relevo, res


def fundir_imagens_v1(imagem_satelite_rgb, imagem_relevo):
    """
    [FUNÇÃO OBSOLETA - MANTIDA APENAS PARA REGISTRO HISTÓRICO] (Mostrar pro professor a evolução do código)

    Esta foi a primeira abordagem de fusão, que substitui o canal de brilho (V) da imagem de satélite diretamente pelo relevo normalizado.
    Função meramente visual para facilitar a compreensão espacial da área observada, mesclando as imagens de relevo e satélite e resultando numa imagem pseudo-3D sombreada

    Problema: Causa distorção e perda de fidelidade nas cores originais.
    Substituída por: fundir_imagens_v2_3D (que usa Hillshade).
    """
    print("Criando visualização 3D (fusão de satélite e relevo)...")

    relevo_normalizado = cv2.normalize(imagem_relevo, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) #Normaliza o relevo para ser o novo canal 0-255 de brilho, o V do HSV [Hue (Matiz), Saturation, Value (Brilho)]
    imagem_hsv = cv2.cvtColor(imagem_satelite_rgb, cv2.COLOR_RGB2HSV) #Converte a imagem de satélite de RGB para HSV, ideal para alterar a luminosidade sem afetar as cores da imagem
    h, s, v = cv2.split(imagem_hsv) #Separa os 3 canais do HSV
    imagem_fundida_hsv = cv2.merge([h, s, relevo_normalizado]) #Reconstrói a imagem a partir da substituição do V (brilho) original pelo V obtido a partir da normalização do relevo
    imagem_final_3d = cv2.cvtColor(imagem_fundida_hsv, cv2.COLOR_HSV2RGB) #Converte a imagem de volta para RGB para visualização, pois é o padrão para bibliotecas como Matplotlib
    
    print("Visualização 3D criada com sucesso.")
    return imagem_final_3d #Retorna a matriz da imagem com o "efeito 3D" processado


def fundir_imagens_v2_3D(imagem_satelite_rgb, imagem_relevo, angulo_sol=315.0, elevacao_sol=45.0):
    """
    Cria uma visualização 3D realista modulando o brilho da imagem de satélite
    com um mapa de relevo sombreado (Hillshade) para preservar as cores.
    """
    print("Criando visualização 3D final (Método de Modulação HSV)...")

    # 1. Cálculo do Hillshade (esta parte está correta e continua igual)
    relevo_float = imagem_relevo.astype(float)
    x, y = np.gradient(relevo_float)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuth_rad = np.deg2rad(angulo_sol)
    altitude_rad = np.deg2rad(elevacao_sol)

    shaded = (np.sin(altitude_rad) * np.sin(slope) +
              np.cos(altitude_rad) * np.cos(slope) * np.cos((azimuth_rad - np.pi / 2.) - aspect))

    hillshade_normalizado = cv2.normalize(shaded, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # 2. Fusão por Modulação de Brilho
    # Converte a imagem de satélite para HSV
    imagem_hsv = cv2.cvtColor(imagem_satelite_rgb, cv2.COLOR_RGB2HSV)

    # Separa os canais H (Cor), S (Saturação) e V (Brilho)
    h, s, v = cv2.split(imagem_hsv)

    # --- INÍCIO DA CORREÇÃO ---
    # Multiplicamos o brilho original (V) pelo mapa de sombras (hillshade).
    # Para isso, ambos precisam ser do tipo float.
    v_float = v.astype(float)

    # O novo brilho é o brilho original modulado pelas sombras.
    # Adicionamos uma "luz ambiente" para que as sombras não fiquem totalmente pretas.
    luz_ambiente = 0.5
    v_novo = np.clip((v_float * (hillshade_normalizado + luz_ambiente)), 0, 255)

    # Converte de volta para o formato de imagem uint8
    v_novo = v_novo.astype(np.uint8)
    # --- FIM DA CORREÇÃO ---

    # Junta os canais originais de Cor e Saturação com o NOVO canal de Brilho modulado
    imagem_fundida_hsv = cv2.merge([h, s, v_novo])

    # Converte a imagem HSV fundida de volta para RGB para visualização
    imagem_final_rgb = cv2.cvtColor(imagem_fundida_hsv, cv2.COLOR_HSV2RGB)

    print("Visualização 3D criada com sucesso.")
    return imagem_final_rgb


def criar_mapa_agua_v1(imagem_satelite_rgb):
    """
    [V1 - MÉTHODO SIMPLES] Cria um mapa de água usando um filtro de cor
    azul no espaço de cores HSV. Rápido, mas pode gerar falsos positivos.
    """
    print("Gerando mapa de água (v1 - Método HSV)...")
    imagem_hsv = cv2.cvtColor(imagem_satelite_rgb, cv2.COLOR_RGB2HSV)
    limite_inferior_azul = np.array([90, 50, 50])
    limite_superior_azul = np.array([130, 255, 255])
    mascara_agua = cv2.inRange(imagem_hsv, limite_inferior_azul, limite_superior_azul)
    return mascara_agua


def criar_mapa_agua_v2(imagem_satelite_rgb, limiar_intensidade=100):
    """
    [V2 - MÉTHODO ROBUSTO] Cria um mapa de água usando regras baseadas na
    reflectância da água nos canais RGB (água absorve mais vermelho).
    """
    print("Gerando mapa de água (v2 - Método RGB)...")
    imagem_bgr = cv2.cvtColor(imagem_satelite_rgb, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(imagem_bgr)

    intensidade_media = (b.astype(float) + g.astype(float) + r.astype(float)) / 3
    mascara_intensidade = intensidade_media < limiar_intensidade

    mascara_azulada = b > r
    mascara_esverdeada = g > r

    mascara_combinada = mascara_intensidade & mascara_azulada & mascara_esverdeada

    mapa_agua_final = mascara_combinada.astype(np.uint8) * 255
    return mapa_agua_final


def criar_mapa_agua_v3_ajustavel(imagem_satelite_rgb, limiar_intensidade, diferenca_azul, diferenca_verde):
    """
    [V3 - AJUSTÁVEL] Cria um mapa de água com parâmetros de sensibilidade personalizáveis.
    """
    # Esta função não imprime nada para não poluir a saída do widget interativo
    imagem_bgr = cv2.cvtColor(imagem_satelite_rgb, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(imagem_bgr)

    # REGRA 1: Intensidade
    intensidade_media = (b.astype(float) + g.astype(float) + r.astype(float)) / 3
    mascara_intensidade = intensidade_media < limiar_intensidade

    # REGRA 2: "Azulidade"
    # Usamos np.subtract para evitar problemas com overflow de uint8 ao subtrair
    delta_azul_vermelho = np.subtract(b, r, dtype=np.int16)
    mascara_azulada = delta_azul_vermelho > diferenca_azul

    # REGRA 3: "Verdura" (em relação ao vermelho)
    delta_verde_vermelho = np.subtract(g, r, dtype=np.int16)
    mascara_esverdeada = delta_verde_vermelho > diferenca_verde

    mascara_combinada = mascara_intensidade & mascara_azulada & mascara_esverdeada

    mapa_agua_final = mascara_combinada.astype(np.uint8) * 255
    return mapa_agua_final


def criar_mapa_agua_v4_ndwi(banda_verde_raw, banda_nir_raw, limiar=0.0):
    """
    Calcula o Índice de Água por Diferença Normalizada (NDWI) e cria
    uma máscara binária de água com base em um limiar.

    Args:
        banda_verde_raw (np.array): Matriz da banda Verde (dados brutos, alinhados).
        banda_nir_raw (np.array): Matriz da banda Infravermelha (dados brutos, alinhados).
        limiar (float): O valor de corte para o NDWI. Pixels com NDWI > limiar
                        serão considerados água. O padrão é 0.0.

    Returns:
        tuple: Uma tupla contendo (mapa_agua, ndwi), onde 'mapa_agua' é a
               máscara binária final (uint8) e 'ndwi' é o mapa de índice (float).
    """
    print(f"Calculando NDWI com limiar de {limiar}...")

    # Converte para float para o cálculo
    verde = banda_verde_raw.astype(float)
    nir = banda_nir_raw.astype(float)

    # Calcula o NDWI, evitando divisão por zero
    numerador = verde - nir
    denominador = verde + nir
    ndwi = np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador != 0)

    # Cria a máscara de água com o limiar
    mascara_ndwi = ndwi > limiar
    mapa_agua = mascara_ndwi.astype(np.uint8) * 255

    print("Mapa de Hidrografia (NDWI) criado com sucesso!")
    return mapa_agua, ndwi


def criar_mapa_declividade(imagem_relevo, resolucao_pixel, nodata_value, limiar_graus=15.0):
    """
    [VERSÃO FINAL ROBUSTA] Calcula a declividade (slope) do terreno de forma
    cientificamente correta, ignorando os valores 'nodata'.
    """
    print(f"Calculando mapa de declividade com limiar de {limiar_graus}°...")

    # 1. Cria uma máscara para identificar onde estão os dados válidos
    mascara_dados_validos = imagem_relevo != nodata_value

    # 2. Cria um array para guardar o resultado do slope, preenchido com 0
    slope_em_graus = np.zeros_like(imagem_relevo, dtype=float)

    # 3. Calcula o gradiente APENAS para os dados válidos
    # Para isso, criamos uma cópia do relevo e substituímos o nodata por um valor neutro (interpolação)
    relevo_copia = imagem_relevo.astype(float)
    relevo_copia[~mascara_dados_validos] = np.mean(relevo_copia[mascara_dados_validos])  # Preenche com a média

    gy, gx = np.gradient(relevo_copia, resolucao_pixel)

    # Calcula a declividade em graus
    slope_calculado = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))

    # 4. Aplica o resultado de volta no nosso array de slope, APENAS nos locais válidos
    slope_em_graus[mascara_dados_validos] = slope_calculado[mascara_dados_validos]

    # 5. Aplica o limiar para criar a máscara final de áreas íngremes
    mascara_declividade = slope_em_graus > limiar_graus
    mapa_final = mascara_declividade.astype(np.uint8) * 255

    print("Mapa de declividade criado com sucesso.")
    return mapa_final, slope_em_graus


def criar_mapa_declividade_visual(imagem_relevo, limiar_indice):
    """
    [VERSÃO VISUAL DEFINITIVA] Calcula um "índice de inclinação" relativo,
    otimizado para a visualização, e cria a máscara binária.
    """
    # 1. Calcula o gradiente simples, que já sabemos que gera um bom contraste visual.
    gy, gx = np.gradient(imagem_relevo.astype(float))
    indice_inclinacao = np.sqrt(gx ** 2 + gy ** 2)

    # 2. Aplica o limiar do slider para criar a máscara binária.
    mascara = indice_inclinacao > limiar_indice
    mapa_binario_final = mascara.astype(np.uint8) * 255

    # Retorna o mapa binário e o mapa de índice para a visualização.
    return mapa_binario_final, indice_inclinacao


def criar_visualizacao_overlay(imagem_base_rgb, mascara, cor_rgb=(255, 0, 0), alpha=0.5):
    """
    [VERSÃO FINAL ESTÁVEL] Cria uma visualização sobrepondo uma máscara colorida
    semi-transparente sobre uma imagem base.
    """
    # 1. Cria uma cópia da imagem para não modificar a original
    imagem_final = imagem_base_rgb.copy()

    # 2. Cria uma imagem de cor sólida (ex: toda vermelha)
    overlay_color = np.full(imagem_base_rgb.shape, cor_rgb, dtype=np.uint8)

    # 3. Mistura a imagem base com a imagem de cor sólida para criar o efeito de transparência
    imagem_misturada = cv2.addWeighted(imagem_final, 1 - alpha, overlay_color, alpha, 0)

    # 4. Usa a máscara como um "estêncil" para aplicar a mistura
    # Garante que a máscara tenha 3 dimensões para funcionar com a imagem colorida
    mascara_3d = mascara[:, :, np.newaxis]

    # Onde a máscara for branca (>0), usa a imagem misturada. Senão, mantém a imagem original.
    imagem_com_overlay = np.where(mascara_3d > 0, imagem_misturada, imagem_final)

    return imagem_com_overlay

""" ---------- ADICIONADO DO OUTRO CÓDIGO A PARTIR DAQUI ---------- """

def criar_mapa_vegetacao_ndvi(banda_vermelha_raw, banda_nir_raw, limiar=0.4):
    """
    [FEATURE 3] Calcula o Índice de Vegetação por Diferença Normalizada (NDVI)
    e cria um mapa binário que identifica áreas de vegetação densa.

    O NDVI é um indicador robusto da saúde e densidade da vegetação. A fórmula é:
    NDVI = (NIR - Vermelho) / (NIR + Vermelho).
    Valores altos (próximos de 1) indicam vegetação densa, enquanto valores baixos
    (próximos de 0 ou negativos) indicam solo, rocha ou água.

    Args:
        banda_vermelha_raw (np.array): Matriz da banda Vermelha (dados brutos).
        banda_nir_raw (np.array): Matriz da banda Infravermelha (dados brutos).
        limiar (float): O valor de corte para o NDVI. Pixels com NDVI > limiar
                        serão considerados vegetação densa. O padrão é 0.4.

    Returns:
        tuple: Uma tupla contendo (mapa_ndvi, mapa_vegetacao).
               - mapa_ndvi (np.array): Mapa com valores de -1 a 1 (o índice científico).
               - mapa_vegetacao (np.array): Mapa binário (0 ou 255) onde 255
                 representa áreas com vegetação ACIMA do limiar.
    """
    print(f"Calculando NDVI com limiar de {limiar}...")

    # 1. Converte as bandas para float para permitir cálculos de divisão.
    vermelho = banda_vermelha_raw.astype(float)
    nir = banda_nir_raw.astype(float)

    # 2. Calcula o NDVI, com segurança para evitar divisão por zero.
    numerador = nir - vermelho
    denominador = nir + vermelho
    mapa_ndvi = np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador != 0)

    # 3. Cria a máscara binária de vegetação com base no limiar.
    mascara_vegetacao = mapa_ndvi > limiar
    mapa_vegetacao = mascara_vegetacao.astype(np.uint8) * 255

    print("Mapa de vegetação (NDVI) criado com sucesso!")
    return mapa_ndvi, mapa_vegetacao


def normalizar_feature(mapa_binario, valor_positivo=1.0, valor_negativo=-1.0):
    """
    [ETAPA 3.1] Converte um mapa de feature binário (0 e 255) em um mapa de
    pontuação normalizado (ex: +1.0 e -1.0).

    Esta função é o primeiro passo para a modelagem do score. Ela traduz as
    features, que estão em "unidades" diferentes (apenas identificando a presença
    ou ausência de uma característica), para uma escala universal de "pontos de
    viabilidade".

    Por padrão, assume-se que o valor 0 no mapa binário é "bom" para a construção
    (ex: sem mata, terreno plano) e o valor 255 é "ruim" (ex: com mata, íngreme).

    Args:
        mapa_binario (np.array): O mapa de feature com valores 0 e 255.
        valor_positivo (float): A pontuação a ser atribuída às áreas "boas" (onde o mapa é 0).
        valor_negativo (float): A pontuação a ser atribuída às áreas "ruins" (onde o mapa é 255).

    Returns:
        np.array: Um mapa com o mesmo formato do original, mas com os valores
                  convertidos para a nova escala de pontuação (em float).
    """
    # 1. Cria um "molde" de floats com o mesmo tamanho do mapa original.
    mapa_normalizado = np.zeros(mapa_binario.shape, dtype=float)

    # 2. Aplica as regras de tradução.
    #    Onde o mapa original era 0 (bom), o novo mapa recebe o valor_positivo.
    mapa_normalizado[mapa_binario == 0] = valor_positivo
    #    Onde o mapa original era 255 (ruim), o novo mapa recebe o valor_negativo.
    mapa_normalizado[mapa_binario == 255] = valor_negativo

    return mapa_normalizado



def criar_score_proximidade_agua(mapa_agua, dist_ideal_pixels=5, dist_max_pixels=100):
    """
    [FEATURE 4] Cria um mapa de score baseado na proximidade da água.

    Calcula a distância de cada pixel de terra até a água mais próxima.
    Em seguida, normaliza essa distância em uma pontuação de viabilidade (-1 a +1),
    onde estar perto da água é considerado positivo.

    Args:
        mapa_agua (np.array): O mapa binário de água (255 para água, 0 para terra).
        dist_ideal_pixels (int): Distância em pixels considerada "perfeita" (score +1.0).
        dist_max_pixels (int): Distância máxima em pixels onde a água ainda tem
                               influência positiva. Acima disso, o score se torna negativo.

    Returns:
        tuple: Uma tupla contendo (mapa_distancia, score_proximidade).
               - mapa_distancia (np.array): Mapa com a distância real em pixels para a água.
               - score_proximidade (np.array): Mapa de pontuação normalizado (-1 a +1).
    """
    print("Calculando score de proximidade à água...")

    # 1. A função distanceTransform precisa do inverso do mapa de água.
    #    Ela calcula a distância de cada pixel NÃO-ZERO até o ZERO mais próximo.
    mapa_terra = (mapa_agua == 0).astype(np.uint8)
    mapa_distancia = cv2.distanceTransform(mapa_terra, cv2.DIST_L2, 5)

    # 2. Normaliza a distância para a escala de pontuação.
    #    Esta fórmula mapeia a distância para o intervalo [-1, 1].
    #    - Se distancia <= dist_ideal, score = 1.0
    #    - Se distancia >= dist_max, score tende a -1.0
    score_proximidade = 1 - 2 * ((mapa_distancia - dist_ideal_pixels) / (dist_max_pixels - dist_ideal_pixels))

    # 3. Garante que o score fique exatamente entre -1 e 1 e zera na água.
    score_proximidade = np.clip(score_proximidade, -1.0, 1.0)
    score_proximidade[mapa_agua == 255] = 0  # Zera o score na água

    print("Score de proximidade à água criado com sucesso!")
    return mapa_distancia, score_proximidade
