from earthaccess import search_data
res = search_data(short_name='MODISA_L3m_CHL_NRT', temporal=('2025-09-26','2025-09-27'), count=10)
print(len(res))
for g in res:
    print(g.data_links())

