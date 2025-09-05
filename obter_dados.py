import ee
import geemap
import time
import os
from datetime import datetime

def autenticar_ee(id_projeto):
    """
    Autentica e inicializa a conexão do script com o ID do nosso projeto do Google Cloud na Google Earth Engine.
    """
    try: #Final feliz: teu email já tá salvo na máquina e não precisa logar
        ee.Initialize(project=id_projeto)
    except Exception as e: #Final triste: abre o navegador e loga com o email no Google Cloud
        print("Autenticação necessária...")
        ee.Authenticate()
        ee.Initialize(project=id_projeto)
    print(">>> Conexão com Google Earth Engine estabelecida com sucesso!")


def baixar_dados_da_area(aoi, pasta_mae="outputs"):
    """
    Cria uma pasta de sessão única com timestamp e baixa os dados de satélite e
    relevo para dentro dela. Retorna o caminho da pasta da sessão.
    """
    #Gera um timestamp para garantir nomes de arquivo únicos e não sobrescrever downloads anteriores
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Formato: ANO-MES-DIA_HORA-MINUTO-SEGUNDO

    #Cria a pasta da sessão
    pasta_sessao = os.path.join(pasta_mae, timestamp)
    os.makedirs(pasta_sessao, exist_ok=True)
    print(f"Sessão de análise iniciada. Arquivos serão salvos em: '{pasta_sessao}'")

    # 3. Define os nomes dos arquivos COM o timestamp, DENTRO da pasta da sessão
    nome_base_satelite = f"{timestamp}_satelite"
    nome_base_relevo = f"{timestamp}_relevo"
    arquivo_satelite = os.path.join(pasta_sessao, f"{nome_base_satelite}.tif")
    arquivo_relevo = os.path.join(pasta_sessao, f"{nome_base_relevo}.tif")

    #PARTE A: IMAGEM DE SATÉLITE
    if os.path.exists(arquivo_satelite): #Evita arquivos repetidos
        print(f"[INFO] O arquivo '{arquivo_satelite}' já existe.")
    else:
        print("\n--- ETAPA 1: Imagem de Satélite ---")
        imagem_satelite = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') #Escolhe o satélite Sentinel-2 como o "acervo" das imagens
            .filterBounds(aoi).filterDate('2024-01-01', '2025-09-01') #Desse acervo, quero somente imagens da minha área de interesse, capturadas a partir de 2024
            .sort('CLOUDY_PIXEL_PERCENTAGE').first().clip(aoi) #Organiza as imagens pela porcentagem de "céu limpo" e pega a melhor, garantindo que não chegue uma imagem com uma nuvem gigante cobrindo tudo que queríamos ver
        )
        task = ee.batch.Export.image.toDrive( #Protocola o "pedido" de download da imagem, com os requisitos a seguir:
            image=imagem_satelite.select(['B4', 'B3', 'B2']), #Pega a imagem que acabamos de filtrar e seleciona somente as bandas R, G e B
            description=f"satelite_{timestamp}", #O nome do "pedido" dentro do Earth Engine
            folder='Oraculum_Data_Export', #A pasta de destino no drive
            fileNamePrefix=nome_base_satelite, #O nome do o arquivo .tif de destino
            scale=10, region=aoi, fileFormat='GeoTIFF' #Cada pixel corresponde a 10m² no mundo real, define o formato de destino como GeoTIFF
        )
        task.start() #Começa a processar nosso pedido
        start_time = time.time() #Guarda a hora atual
        while True:
            status = task.status() #Get no status atual do pedido (READY, RUNNING, COMPLETED ou FAILED)
            state = status['state'] #Imprime o status atual
            elapsed_time = int(time.time() - start_time) #Imprime o tempo que o pedido levou para ser processado
            print(f"Exportando para o Google Drive... (Status: {state}, Tempo: {elapsed_time}s)")
            if state in ['COMPLETED', 'FAILED']: #Condições de parada do while
                if state == 'COMPLETED':
                    print("\n[SUCESSO] Imagem de satélite exportada para o Google Drive!")
                    print(">>> AÇÃO NECESSÁRIA: Baixe o arquivo e coloque na pasta do projeto. <<<")
                else:
                    print(f"\n[ERRO] A exportação falhou: {status.get('error_message', 'Sem detalhes')}")
                break
            time.sleep(30) #Aguarda 30 segundos para exibir o status atual novamente

    #PARTE B: IMAGEM DE RELEVO
    if os.path.exists(arquivo_relevo): #Evita arquivos repetidos
        print(f"\n[INFO] O arquivo de relevo já existe nesta pasta.")
    else:
        print("\n--- ETAPA 2: Imagem de Relevo ---")
        imagem_relevo = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(aoi) #Escolhe o mapa de elevação NASA DEM para puxar os dados de relevo da nossa AOI
        geemap.ee_export_image( #Como o arquivo de relevo, com resolução de 30m²/pixel, é muito mais leve que a imagem de satélite, dá pra baixar direto pro computador ao invés de ter que upar para o Drive
            imagem_relevo, filename=arquivo_relevo, #Define o nome do arquivo a ser salvo (já com timestamp)
            scale=30, region=aoi, file_per_band=False #Cada pixel corresponde a 30m² no mundo real, "file_per_band=False" especifica que queremos um arquivo único com todas as bandas de informação, não um arquivo diferente para cada banda
        )
        print(f"[SUCESSO] Arquivo de relevo baixado diretamente.")

    # 3. Retorna o CAMINHO DA PASTA da sessão, que é a informação mais importante agora
    return arquivo_satelite, arquivo_relevo