import numpy as np
import cv2
import rasterio
from rasterio.warp import reproject, Resampling


'''
Uma imagem de satélite, ao invés de guardar as cores de cada pixel em 8 bits (r,g,b) (0-255) com uma faixa dinâmica baixa, guarda os dados com 16 bits (0-65535), uma faixa dinâmica alta (HDR), se tentarmos exibir a 
imagem primitiva, como as telas de notebook e celular geralmente são configuradas para 0-255, seria exibida uma imagem preta com baixíssimo contraste, por isso, essa função pega a faixa enorme e converte de forma 
inteligente para 0-255, realçando os detalhes visíveis ignorando valores extremos (pixels muito escuros ou reflexos muito brilhantes, que são como "ruído", e foca somente na faixa de dados com a informação relevante 
para a análise do terreno.
'''
def ajustar_contraste_canal(canal):
    p2, p98 = np.percentile(canal, (2, 98)) #Guarda o "limiar" dos 2% mais escuros e 2% mais claros dos pixels (basicamente isola os intervalos de possíveis outliers)
    canal_cortado = np.clip(canal, p2, p98) #"Limpa" a imagem, fazendo com que qualquer pixel mais escuro que p2 = p2 e qualquer pixel mais claro que p98 = p98 ("normaliza" os outliers)
    canal_ajustado = cv2.normalize(canal_cortado, None, 0, 255, cv2.NORM_MINMAX) #(p2,p98) = (0,255), todos os outros valores são distribuídos nesse intervalo, fazendo com que toda a faixa de dados fique visível
    canal_final = canal_ajustado.astype(np.uint8) #converte o tipo de dados da matriz já ajustada para dados inteiros de 8 bits, agora sim com o intervalo 0-255 esperado por bibliotecas como matplotlib
    return canal_final


'''
Imagem de Satélite: Colorida, alta resolução e "na diagonal" (não necessariamente está alinhada com o norte) -> Coordenadas em UTM (medido em metros)
Imagem de Relevo: Elevação, resolução menor e "reta"                                                         -> Coordenadas em Latitude e Longitude (medido em graus)

Se tentássemos sobrepor as duas matrizes de pixels, não ia funcionar, pois o mesmo pixel em cada matriz pode corresponder a dois pontos geográficos distintos, assim, precisamos "alinhar" as duas imagens.
Usando a imagem de satélite como referência, podemos reprojetar a imagem de relevo (modificando escala, rotação e orientação para "encaixar" na de satélite), resolvendo assim o problema das coordenadas divergentes.
'''
def alinhar_imagens(caminho_satelite, caminho_relevo):
    print("Iniciando carregamento e reprojeção para alinhamento...")
    with rasterio.open(caminho_satelite) as src_sat: #Abre o arquivo GeoTIFF da imagem de satélite
        profile_sat = src_sat.profile #Lê os metadados da imagem (sistema de coordenadas, altura/largura, escala etc) e define como o perfil padrão (molde)
        sat_data_raw = src_sat.read() #Lê os dados dos pixels da imagem e armazena tudo em um array

    relevo_reprojetado = np.zeros((profile_sat['height'], profile_sat['width']), dtype=profile_sat['dtype']) #Cria uma matriz nula (preenchida com zeros) 'np.zeros' como uma tela de pintura em branco com o molde construído a partir da imagem de satélite
    with rasterio.open(caminho_relevo) as src_rel: #Abre o arquivo GeoTIFF da imagem de relevo
        reproject( #Inicializa a estrutura do processo de "reforma" da imagem
            source=rasterio.band(src_rel, 1), #Define a banda 1 (de elevação) como a fonte original dos dados que vamos querer "reformar"
            destination=relevo_reprojetado, #Define a "tela de pintura em branco" como o destino dos dados "reformados"
            src_transform=src_rel.transform, #"Get" no perfil da imagem de relevo
            src_crs=src_rel.crs, #"Get" no tipo de coordenada da imagem de relevo, nesse caso Lat/Lon
            dst_transform=profile_sat['transform'], #"Set" no perfil de destino (satélite)
            dst_crs=profile_sat['crs'], #"Set" no tipo de coordenada da imagem de destino, nesse caso UTM (metros)
            resampling=Resampling.bilinear #Usa interpolação bilinear como técnica de "transformação" para um resultado suave
        )
    
    print("Reprojeção concluída!")
    
    imagem_satelite_bruta_cv = np.transpose(sat_data_raw, (1, 2, 0)) #A biblioteca Rasterio lê como (Canais, Altura, Largura), então reorganizamos para (Altura, Largura, Canais) que é o padrão do OpenCV
    r, g, b = cv2.split(imagem_satelite_bruta_cv) #Separa os 3 canais de cor (red, green, blue) para poder ajustar o contraste de cada um individualmente
    r_ajustado = ajustar_contraste_canal(r) #Chama a função que criamos para ajustar o intervalo do canal red para 0-255
    g_ajustado = ajustar_contraste_canal(g) #Chama a função que criamos para ajustar o intervalo do canal green para 0-255
    b_ajustado = ajustar_contraste_canal(b) #Chama a função que criamos para ajustar o intervalo do canal blue para 0-255
    imagem_satelite_final = cv2.merge([r_ajustado, g_ajustado, b_ajustado]) #Junta novamente os 3 canais numa imagem RGB final
    
    imagem_relevo_final = relevo_reprojetado #Só pra deixar mais bonito quando for chamar
    
    print("Processamento finalizado. Retornando imagens alinhadas.")
    return imagem_satelite_final, imagem_relevo_final #Retorna as matrizes da imagem de satélite (com o contraste ajustado) e da imagem de relevo ("alinhada" com a de satélite)


'''
Função meramente visual para facilitar a compreensão espacial da área observada, mesclando as imagens de relevo e satélite e resultando numa imagem pseudo-3D sombreada
'''
def fundir_imagens(imagem_satelite_rgb, imagem_relevo):
    print("Criando visualização 3D (fusão de satélite e relevo)...")

    relevo_normalizado = cv2.normalize(imagem_relevo, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) #Normaliza o relevo para ser o novo canal 0-255 de brilho, o V do HSV [Hue (Matiz), Saturation, Value (Brilho)]
    imagem_hsv = cv2.cvtColor(imagem_satelite_rgb, cv2.COLOR_RGB2HSV) #Converte a imagem de satélite de RGB para HSV, ideal para alterar a luminosidade sem afetar as cores da imagem
    h, s, v = cv2.split(imagem_hsv) #Separa os 3 canais do HSV
    imagem_fundida_hsv = cv2.merge([h, s, relevo_normalizado]) #Reconstrói a imagem a partir da substituição do V (brilho) original pelo V obtido a partir da normalização do relevo
    imagem_final_3d = cv2.cvtColor(imagem_fundida_hsv, cv2.COLOR_HSV2RGB) #Converte a imagem de volta para RGB para visualização, pois é o padrão para bibliotecas como Matplotlib
    
    print("Visualização 3D criada com sucesso.")
    return imagem_final_3d #Retorna a matriz da imagem com o "efeito 3D" processado