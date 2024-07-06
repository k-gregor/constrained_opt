# Abies alba, Pinus sylvetris, Pinus halepensis, Juniperus oxycedrus, Picea abies
# Weisstanne, Waldkiefer, Aleppo-Kiefer, Stechwacholder, Gemeine Fichte
# EU silver fir, scots pine, Aleppo pine, Cedar/Cade, Norway spruce
ne_pfts = ['Abi_alb', 'Pin_syl', 'Pin_hal', 'Jun_oxy', 'Pic_abi']
nd_pfts = ['Lar_dec'] # Laerche
be_pfts = ['Que_ile'] # Immergruene Eiche
# Ulmus glabra, Tilia Cordata, Betula pendula, Betula pubescens, Corylus avellana, Carpinus betulus, Fagus sylvatica, Fraxinus excelsior, Quercus pubescens, Quercus robur
#           Ulme,  Linde, Zitterpappel/Espe, Weissbirke, Moorbirke,  Hasel, Hainbuche, Rotbuche,  Esche,     Flaumeiche, Deutsche eiche
#          elm, small-leaved lime, EU aspen, silver birch, downy birch, hazel, hornbeam, EU beech, ash, downy oak, European/common Oak
bd_pfts = ['Ulm_gla', 'Til_cor', 'Pop_tre', 'Bet_pen', 'Bet_pub', 'Cor_ave', 'Car_bet', 'Fag_syl', 'Fra_exc', 'Que_pub', 'Que_rob']
shrub_pfts = ['BES', 'MRS', 'Que_coc']
grass_pfts = ['C3_gr']
all_mats_no_grass = ['Abi_alb','BES','Bet_pen','Bet_pub','Car_bet','Cor_ave','Fag_syl','Fra_exc','Jun_oxy','Lar_dec','MRS','Pic_abi','Pin_syl','Pin_hal','Pop_tre','Que_coc','Que_ile','Que_pub','Que_rob','Til_cor','Ulm_gla']
all_mats_only_trees = ['Abi_alb','Bet_pen','Bet_pub','Car_bet','Cor_ave','Fag_syl','Fra_exc','Jun_oxy','Lar_dec','Pic_abi','Pin_syl','Pin_hal','Pop_tre','Que_ile','Que_pub','Que_rob','Til_cor','Ulm_gla']


pft_to_forest_type_map = {}
for pft in ne_pfts:
    pft_to_forest_type_map[pft] = "NE"
for pft in nd_pfts:
    pft_to_forest_type_map[pft] = "ND"
for pft in be_pfts:
    pft_to_forest_type_map[pft] = "BE"
for pft in bd_pfts:
    pft_to_forest_type_map[pft] = "BD"
for pft in shrub_pfts:
    pft_to_forest_type_map[pft] = "Shrub"
for pft in grass_pfts:
    pft_to_forest_type_map[pft] = "Grass"
