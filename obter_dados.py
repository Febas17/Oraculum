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
    Cria uma pasta de sessão com timestamp, e baixa os dados de satélite e relevo
    para dentro dela com nomes de arquivo também com timestamp.
    Usa um méthodo de download dinâmico (direto ou via Drive).
    Retorna os caminhos completos para os arquivos criados.
    """
    # 1. Gera um timestamp único para a sessão inteira
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Gerando arquivos para a sessão: {timestamp}")

    # 2. Cria a pasta da sessão
    pasta_sessao = os.path.join(pasta_mae, timestamp)
    os.makedirs(pasta_sessao, exist_ok=True)

    # 3. Define os nomes dos arquivos COM o timestamp, DENTRO da pasta da sessão
    nome_base_satelite = f"{timestamp}_satelite"
    nome_base_relevo = f"{timestamp}_relevo"
    arquivo_satelite = os.path.join(pasta_sessao, f"{nome_base_satelite}.tif")
    arquivo_relevo = os.path.join(pasta_sessao, f"{nome_base_relevo}.tif")

    # --- PARTE A: IMAGEM DE SATÉLITE (DOWNLOAD DINÂMICO) ---
    if os.path.exists(arquivo_satelite):
        print(f"[INFO] O arquivo de satélite '{arquivo_satelite}' já existe.")
    else:
        print("\n--- ETAPA 1: Imagem de Satélite ---")
        imagem_satelite = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi).filterDate('2024-01-01', '2025-09-01')
            .sort('CLOUDY_PIXEL_PERCENTAGE').first().clip(aoi)
        )

        try:
            # TENTATIVA 1: Download Direto
            print("Tentando download direto (método rápido)...")
            geemap.ee_export_image(
                imagem_satelite.select(['B4', 'B3', 'B2', 'B8']),
                filename=arquivo_satelite, scale=10, region=aoi
            )
            print(f"[SUCESSO] Arquivo de satélite salvo diretamente em: {arquivo_satelite}")

        except ee.EEException as e:
            # Se falhar por tamanho, ativa o Plano B
            if 'Total request size' in str(e):
                print("[AVISO] Arquivo muito grande. Ativando Plano B: exportação via Google Drive.")
                task = ee.batch.Export.image.toDrive(
                    image=imagem_satelite.select(['B4', 'B3', 'B2', 'B8']),
                    description=nome_base_satelite,  # Usa o nome com timestamp
                    folder='Oraculum_Data_Export',
                    fileNamePrefix=nome_base_satelite,  # Usa o nome com timestamp
                    scale=10, region=aoi, fileFormat='GeoTIFF'
                )
                task.start()
                # (A lógica de monitoramento da tarefa entra aqui, como antes)
                # ...
            else:
                # Se for outro erro, exibe
                print(f"Ocorreu um erro inesperado no Earth Engine: {e}")
                raise e

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