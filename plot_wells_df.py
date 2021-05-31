from folium.map import FeatureGroup

# Hi Scott


def plot_wells_df(df, lat, long, name, *vars, mapname='Map1'):
    """
    Plots wells using folium colored by various attributes

    Args: 
        df(str)- pandas dataframe 
        lat(str) - column name for lattitude values
        long(str) - column name for longitude values
        name(str) - column name for well name
        attr(str) - column name for attribute to color well locations by
        attr(str optional) - column name for secondary attribute to color well locations by
        mapname(str optional) - name for the output map
    """


    # Import libraries
    import folium
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    print(df.head())
    # Create 5 lists for the surface coordinates, well UWI, elevation, and td
    surf_lat = list(df[lat])
    surf_long = list(df[long])
    name = list(df[name])
    
     # Let's center the map by calculating the average lat and long
    map_lat = np.mean(surf_lat)
    map_long = np.mean(surf_long)

    # Create a popup message with information about the well and a link
    html = """<h4>Duvernay Well Information:</h4>
    Name: <a href="https://www.google.com/search?q={} Duverney" target="_blank">{}</a><br>
    Value: {} m
    """
    map = folium.Map(location=[map_lat,map_long],tiles=None,zoom_start=5)
    base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
    folium.TileLayer(tiles='Stamen Terrain').add_to(base_map)
    base_map.add_to(map)

    ind = 0
    # create markers for attributes
    for var in vars:

        attr = list(df[var])
        # calculate min/max for attr
        attr_min = np.nanmin(attr)
        attr_max = np.nanmax(attr)
        print(var + ' min = ' + str(attr_min) + ', max = ' + str(attr_max))
    
        # Create a color table for the circle marker
        color = list(np.empty(len(attr)))
        crange = np.linspace(attr_min,attr_max,5)
        ind = 0
        for a in attr:
            if a <= crange[0]:
                color[ind] = 'blue'
                ind += 1
            elif crange[0]< a <=crange[1]:
                color[ind] = 'green'
                ind += 1
            elif crange[1]< a <=crange[2]:
                color[ind] = 'yellow'
                ind += 1
            elif crange[2]< a <=crange[3]:
                color[ind] = 'orange'
                ind += 1
            elif crange[3]< a:
                color[ind] = 'red'
                ind += 1
            else:
                color[ind] = 'grey'
                ind += 1
        
        
        fg=folium.FeatureGroup(var,overlay=False)

        for lt, ln, nm, a1, c in zip(surf_lat,surf_long,name,attr,color):
            iframe = folium.IFrame(html=html.format(nm, nm, str(a1)), width=200, height=100)
            folium.CircleMarker(location=[lt,ln], popup=folium.Popup(iframe), radius=4, color='black', 
                                            opacity=0.5, fill_color=c, fill_opacity=0.7).add_to(fg).add_to(fg)
        fg.add_to(map)
    folium.LayerControl().add_to(map)    
                                
    mapname += '.html'
    map.save(mapname)
    
    return print(mapname + ' created successfully')