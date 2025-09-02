import ee
import geemap
import time
import os

# --- CONFIGURAÇÃO ---
ID_PROJETO_GOOGLE = 'oraculum-eseg'

# Coordenadas do polígono da área de estudo
COORDENADAS = [
    [-47.06466, -20.21627],  # Canto Superior Esquerdo (CSE)
    [-47.04290, -20.30439],  # Canto Inferior Esquerdo (CIE)
    [-46.94930, -20.28505],  # Canto Inferior Direito (CID)
    [-46.97212, -20.19713],  # Canto Superior Direito (CSD)
]

# Nomes dos arquivos de saída
ARQUIVO_SATELITE = 'satelite_oraculum.tif'
ARQUIVO_RELEVO = 'relevo_oraculum.tif'


# --- INICIALIZAÇÃO DO EARTH ENGINE ---
try:
    ee.Initialize(project=ID_PROJETO_GOOGLE)
except Exception as e: #Da primeira vez pode pedir pra logar:
    print("Autenticação necessária...")
    ee.Authenticate()
    ee.Initialize(project=ID_PROJETO_GOOGLE)

print(">>> Conexão com Google Earth Engine estabelecida com sucesso!")
aoi = ee.Geometry.Polygon(COORDENADAS)


# --- ETAPA DE DOWNLOAD ---
# PARTE 1: IMAGEM DE SATÉLITE (Arquivo Grande -> Exportar para Google Drive)
if os.path.exists(ARQUIVO_SATELITE):
    print(f"\n[INFO] O arquivo '{ARQUIVO_SATELITE}' já existe. Etapa do satélite pulada.")
else:
    print("\n--- ETAPA 1: Imagem de Satélite ---")
    print("Procurando a melhor imagem de satélite (Sentinel-2 Harmonized)...")

    imagem_satelite = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate('2024-01-01', '2025-09-01')
        .sort('CLOUDY_PIXEL_PERCENTAGE')
        .first()
        .clip(aoi)
    )

    print("Iniciando exportação da imagem para o Google Drive. Isso pode levar vários minutos.")

    task = ee.batch.Export.image.toDrive(
        image=imagem_satelite.select(['B4', 'B3', 'B2']),  # Bandas de cor real
        description='satelite_oraculum_export',
        folder='Oraculum_Data_Export',
        fileNamePrefix=ARQUIVO_SATELITE.replace('.tif', ''),
        scale=10,
        region=aoi,
        fileFormat='GeoTIFF'
    )
    task.start()

    start_time = time.time()
    while True:
        status = task.status()
        state = status['state']
        elapsed_time = int(time.time() - start_time)
        print(f"Status da exportação: {state} (Tempo decorrido: {elapsed_time}s)")

        if state == 'COMPLETED':
            print("\n[SUCESSO] Imagem de satélite exportada para o seu Google Drive!")
            print(">>> AÇÃO NECESSÁRIA: <<<")
            print("1. Vá ao seu Google Drive, encontre a pasta 'Oraculum_Data_Export'.")
            print(f"2. Baixe o arquivo '{ARQUIVO_SATELITE}'.")
            print("3. Mova o arquivo para a pasta deste projeto no seu computador.")
            break
        elif state == 'FAILED':
            print(f"\n[ERRO] A exportação falhou. Mensagem: {status.get('error_message', 'Sem detalhes')}")
            break

        time.sleep(30)

# PARTE 2: IMAGEM DE RELEVO (Arquivo Menor -> Download Direto)
if os.path.exists(ARQUIVO_RELEVO):
    print(f"\n[INFO] O arquivo '{ARQUIVO_RELEVO}' já existe. Etapa do relevo pulada.")
else:
    print("\n--- ETAPA 2: Imagem de Relevo ---")
    print("Procurando dados de elevação (NASA DEM)...")

    imagem_relevo = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(aoi)

    print(f"Iniciando download direto do arquivo '{ARQUIVO_RELEVO}'...")
    try:
        geemap.ee_export_image(
            imagem_relevo,
            filename=ARQUIVO_RELEVO,
            scale=30,
            region=aoi,
            file_per_band=False
        )
        print(f"[SUCESSO] Arquivo '{ARQUIVO_RELEVO}' baixado diretamente.")
    except Exception as e:
        print(f"[ERRO] O download direto do relevo falhou: {e}")

# --- FINALIZAÇÃO ---
print("\n>>> Processo finalizado! <<<")
print("Verifique se os dois arquivos .tif estão na pasta do seu projeto antes de prosseguir para o notebook.")
