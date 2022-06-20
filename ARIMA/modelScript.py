from modelHelper import arimaModel
from naiveModel import naiveModel

datapaths = [
'../generated_data2/flat_2 or less people_before 1900.zip',
'../generated_data2/flat_2 or less people_after 1900.zip',
'../generated_data2/flat_3 or more people_before 1900.zip'
#'../generated_data2/flat_3 or more people_after 1900.zip',
#'../generated_data2/house_2 or less people_after 1965.zip',
#'../generated_data2/house_2 or less people_before 1965.zip',
#'../generated_data2/house_3 or more people_before 1965.zip',
#'../generated_data2/house_3 or more people_after 1965.zip'
]

names = [
'flat_2 or less people_before 1900',
'flat_2 or less people_after 1900',
'flat_3 or more people_before 1900'
#'flat_3 or more people_after 1900',
#'house_2 or less people_after 1965',
#'house_2 or less people_before 1965',
#'house_3 or more people_before 1965',
#'house_3 or more people_after 1965'
]

for i in range(3):
	dp = datapaths[i]
	nm = names[i]
	print("######### NAME: " + nm)
	init1 = arimaModel(dp, nm)
	init1.main()
	print('\n')