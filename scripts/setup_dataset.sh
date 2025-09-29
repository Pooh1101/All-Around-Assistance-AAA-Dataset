set -e

mkdir -p bags/AU bags/CUA bags/UDC
mkdir -p frames

# AU
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/HKKJPC" -O bags/AU/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/FIVWIQ" -O bags/AU/02.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/OUKDBJ" -O bags/AU/03.bag

# CUA
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/THBOG6" -O bags/CUA/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/9RUNGG" -O bags/CUA/02.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/2CU9U0" -O bags/CUA/03.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/L5GBBK" -O bags/CUA/04.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/TSP39E" -O bags/CUA/05.bag

# GMU
    # jcMuInEn
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/LLZEQI" -O bags/GMU/jcMuInEn/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/EVJWUZ" -O bags/GMU/jcMuInEn/02.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/ZMKXOR" -O bags/GMU/jcMuInEn/03.bag

    # jcScEn
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/AXNPPS" -O bags/GMU/jcScEn/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/NJHTNC" -O bags/GMU/jcScEn/02.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/WBNV9A" -O bags/GMU/jcScEn/03.bag

    # JcHubSdEn
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/SGHRJO" -O bags/GMU/JcHubSdEn/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/GDX5RX" -O bags/GMU/JcHubSdEn/02.bag

# GTown
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/HYFAT7" -O bags/GTown/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/D2ZP2K" -O bags/GTown/02.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/IDH2CP" -O bags/GTown/03.bag

# GTown2
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/B9IIRU" -O bags/GTown2/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/3H3AQJ" -O bags/GTown2/02.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/RUKGZS" -O bags/GTown2/03.bag

# GWU
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/BUBSPR" -O bags/GWU/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/JVPKUR" -O bags/GWU/02.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/KVD6V0" -O bags/GWU/03.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/GBZB8A" -O bags/GWU/04.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/PWJLSZ" -O bags/GWU/05.bag

# Marymount
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/GOJRNA" -O bags/Marymount/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/I2L2E0" -O bags/Marymount/02.bag

# Nova
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/IFBBBX" -O bags/Nova/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/UZ64IG" -O bags/Nova/02.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/XEZUJM" -O bags/Nova/03.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/GDFNXL" -O bags/Nova/04.bag

# UDC
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/TPKFMC" -O bags/UDC/01.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/EUUSZL" -O bags/UDC/02.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/O6USW7" -O bags/UDC/03.bag
wget "https://DATAVERSE.orc.gmu.edu/api/access/datafile/:persistentId?persistentId=doi:10.13021/ORC2020/JUIW5F/5DQJDL" -O bags/UDC/04.bag

python3 scripts/bag_to_images.py --input bags --output frames

rm -rf bags

echo "[INFO] Done. All frames are stored in 'frames/'"
