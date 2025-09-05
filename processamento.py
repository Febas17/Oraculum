import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling

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
    Imagem de Satélite: Colorida, alta resolução e "na diagonal" (não necessariamente está alinhada com o norte) → Coordenadas em UTM (medido em metros)
    Imagem de Relevo: Elevação, resolução menor e "reta"                                                         → Coordenadas em Latitude e Longitude (medido em graus)

    Se tentássemos sobrepor as duas matrizes de pixels, não ia funcionar, pois o mesmo pixel em cada matriz pode corresponder a dois pontos geográficos distintos,
    assim, precisamos "alinhar" as duas imagens. Usando a imagem de satélite como referência, podemos reprojetar a imagem de relevo (modificando escala, rotação e
    orientação para "encaixar" na de satélite), resolvendo assim o problema das coordenadas divergentes.
    """
    # 1. Define uma grade de destino ORTOGONAL (reta) a partir do satélite
    with rasterio.open(caminho_satelite) as src_sat:
        bounds = src_sat.bounds
        res = src_sat.res[0]

        dst_width = int(np.ceil((bounds.right - bounds.left) / res))
        dst_height = int(np.ceil((bounds.top - bounds.bottom) / res))

        dst_transform = rasterio.transform.from_origin(bounds.left, bounds.top, res, res)

        dst_profile = src_sat.profile.copy()
        dst_profile.update({
            "transform": dst_transform,
            'width': dst_width, 'height': dst_height,
            'nodata': 0  # Define 0 como o valor para pixels vazios no satélite
        })

    # 2. Cria "telas em branco" para receber os dados alinhados
    satelite_alinhado_raw = np.zeros((dst_profile['count'], dst_height, dst_width), dtype=dst_profile['dtype'])
    with rasterio.open(caminho_relevo) as src_rel:
        relevo_alinhado_raw = np.zeros((1, dst_height, dst_width), dtype=src_rel.profile['dtype'])
        # Guarda o valor nodata original do relevo para usarmos depois
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
            dst_nodata=nodata_value_relevo  # Garante que o valor nodata seja mantido
        )
    print("Reprojeção concluída!")

    # 4. Processa o satélite para visualização
    imagem_satelite_final_cv = np.transpose(satelite_alinhado_raw, (1, 2, 0))
    r, g, b = cv2.split(imagem_satelite_final_cv)
    r_ajustado = ajustar_contraste_canal(r)
    g_ajustado = ajustar_contraste_canal(g)
    b_ajustado = ajustar_contraste_canal(b)
    imagem_satelite_ajustada_final = cv2.merge([r_ajustado, g_ajustado, b_ajustado])

    # 5. Prepara o relevo final
    imagem_relevo_final = relevo_alinhado_raw[0]

    print("Processamento finalizado.")
    # 6. Retorna as imagens e o valor nodata do relevo
    return imagem_satelite_ajustada_final, imagem_relevo_final, nodata_value_relevo


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
