import re
import numpy as np

def dms_para_dd(dms_str):
    """
    A desgraça do Google Earth dá as coordenadas DMS então ntj, tem que fazer isso aí.
    Converte uma coordenada em Graus, Minutos, Segundos (DMS) para Graus Decimais (DD).

    Entrada:
        dms_str (str): Coordenada no formato "20°12'58.58\"S" ou similar.

    Saída:
        float: A coordenada em Graus Decimais (negativa para Sul e Oeste).
    """
    partes = re.findall(r'[\d\.]+', dms_str) #Separa todos os trechos do texto que contenham 1 ou mais (+) dígitos (\d) ou pontos (.)   Ex: " 20º14'58.2"S " = ['20','14','58.2']
    if len(partes) != 3: #Verifica se tu não fez merda e digitou a coordenada no formato errado
        raise ValueError("Formato de coordenada inválido. Use algo como \"20°12'58.58\"S\"")

    graus, minutos, segundos = [float(p) for p in partes] #Separa cada elemento do texto já cortado e converte em float para poder fazer cálculos
    decimal = graus + (minutos / 60) + (segundos / 3600) #Fórmula matemática padrão para converter DMS em DD

    if dms_str.upper().endswith('S') or dms_str.upper().endswith('O') or dms_str.upper().endswith('W'): # Converte para negativo se for Sul (S) ou Oeste (W)
        decimal *= -1

    return decimal


def criar_bounding_box(lat_centro, lon_centro, tamanho_km):
    """
    O Google Earth não tem uma função específica para obter um quadrado de x área, nos primeiros dias usávamos a sensacional ideia de arrastar 4 linhas MANUALMENTE
    pelo mapa e SUPOR que era um quadrado perfeito, o que obviamente não era muito otimizado ou inteligente, então criamos uma função que cria um polígono quadrado
    de área personalizável a partir de um ponto central no mapa 😎 agora sim.

    TL:DR - Cria um polígono quadrado de dimensão qualquer (bounding box) ao redor de um ponto central.
    """
    meio_lado = tamanho_km / 2 #Calcula a distância do ponto central até cada borda do quadrado

    delta_lat = meio_lado / 111.32 #Converte a distância em km para graus de latitude
    delta_lon = meio_lado / (111.32 * np.cos(np.radians(lat_centro))) #Converte a distância em km para graus de longitude

    # Calcula os 4 cantos do quadrado
    lat_norte = lat_centro + delta_lat
    lat_sul = lat_centro - delta_lat
    lon_leste = lon_centro + delta_lon
    lon_oeste = lon_centro - delta_lon

    return [ # Retorna os cantos no formato que o GEE espera: [ [lon, lat], [lon, lat], ... ]
        [lon_oeste, lat_sul], #Canto inferior-esquerdo
        [lon_leste, lat_sul], #Canto inferior-direito
        [lon_leste, lat_norte], #Canto superior-direito
        [lon_oeste, lat_norte] #Canto superior-esquerdo
    ]