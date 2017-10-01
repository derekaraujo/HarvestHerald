#! /usr/bin/env python

import folium
import branca
import os
import pandas as pd


raw_subsidies_data_files = [f for f in os.listdir(‘../data') if 
                                         f.startswith('USFarmSubsidiesProducerPayment')]
for i in range(len(raw_subsidies_data_files)):
    if i == 0:
        subsidies_df = pd.read_csv(os.path.join(‘../data', raw_subsidies_data_files[0]))
    else:
        temp_df = pd.read_csv(os.path.join(‘../data', raw_subsidies_data_files[i]))
        subsidies_df = subsidies_df.append(temp_df, ignore_index=True)
del temp_df


subsidies = subsidies_df.groupby(['state_code', 'county_code', 'calendar_year'], as_index=False).sum()

def get_fips_code(row):
    state_code = str(int(row.state_code))
    county_code = str(int(row.county_code))
    while len(county_code) < 3:
        county_code = '0' + county_code
    fips_code = int('%s%s' % (state_code, county_code))
    return fips_code
    
subsidies['FIPS_Code'] = subsidies.apply(get_fips_code, axis=1)


county_geo = os.path.join('..', 'data', 'us_counties_20m_topo.json')

year = 2012
df = subsidies[subsidies.calendar_year == year]

colorscale = branca.colormap.linear.YlGn.scale(0, 1e6)
subsidy_series = df.set_index('FIPS_Code')['transaction_amount']

def style_function(feature):
    num = subsidy_series.get(int(feature['id'][-5:]), None)
    return {
        'fillOpacity': 0.5,
        'weight': 0,
        'fillColor': '#black' if num is None else colorscale(num)
    }

m = folium.Map(
    location=[48, -102],
    tiles='cartodbpositron',
    zoom_start=3,
)

folium.TopoJson(
    open(county_geo),
    'objects.us_counties_20m',
    style_function=style_function
).add_to(m)

colormap = branca.colormap.linear.YlGn.scale(0, 1e6)#.to_step(int(1e4))
colormap.caption = 'Agricultural Subsidies (USD) per County, %s' % year
m.add_child(colormap)

m.save('subsidy_map.html')



