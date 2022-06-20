from home_features_parse import featureParser
import matplotlib.pyplot as plt
import pandas as pd

def plotAvgwithConfInt(df, name):
    str0 = name.split("_")
    htype = str0[0]
    nrppl = str0[1]
    bldera = str0[2]
    #sns.lineplot(x = df.index, y = "avg", data = df)
    plt.plot(df.index, df.avg, c='purple', label = "average consumption")
    plt.fill_between(df.index, df.max_val, color='blue', alpha=.2, label = "Standard Deviation")
    title = htype + "s with " + nrppl + ", built " + bldera
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('average electricity consumption (Watts)')
    xticks = pd.date_range(min(df.index), max(df.index), periods = 5)
    plt.xticks([x.strftime('%Y-%m-%d') for x in xticks])
    filename = name + ".pdf"
    plt.legend()
    plt.savefig('../imgs2/' + filename)
    plt.close()
    plt.cla()
    plt.clf()
    #plt.show()

# flat, 3 or more people, after 1900
home_list0 = [84, 102, 147, 148, 188, 190, 214, 212, 244, 252, 276, 284, 301, 304, 309, 322]
name0 = 'flat_3 or more people_after 1900'
#flat, 2 or less ppl, before 1900
home_list1 = [62, 72, 74, 83, 105, 114, 116, 117, 119, 125, 136, 135, 144, 145, 150, 153, 156, 158, 160, 159, 167, 179, 205, 208, 215, 225, 227, 234, 249, 248, 250, 251, 253, 256, 267, 269, 275, 278, 279, 286, 292, 294, 295, 310, 311, 328, 319, 313, 323, 320]
name1 = 'flat_2 or less people_before 1900'
# flat, 2 or less ppl, after 1900
home_list2 = [47, 59, 67, 70, 75, 71, 77, 80, 91, 94, 100, 106, 118, 121, 128, 137, 141, 143, 149, 165, 157, 162, 189, 193, 194, 197, 213, 216, 221, 209, 226, 237, 239, 238, 241, 257, 258, 262, 272, 274, 285, 289, 288, 300, 302, 317, 321, 332]
name2 = 'flat_2 or less people_after 1900'
# flat, 3 or more ppl, before 1900
home_list3 = [64, 73, 78, 76, 85, 93, 98, 97, 122, 123, 124, 139, 140, 180, 177, 182, 201, 218, 240, 255, 265, 280, 282, 287, 296, 334, 333]
name3 = 'flat_3 or more people_before 1900'
# house, 2 or less ppl, before 1965
home_list4 = [61, 63, 65, 68, 81, 86, 82, 89, 120, 126, 151, 155, 161, 178, 183, 266, 259, 281, 283, 291, 298, 305, 308, 315]
name4 = 'house_2 or less people_before 1965'
# house, 2 or less ppl, after 1965
home_list5 = [88, 90, 101, 107, 169, 163, 175, 184, 186, 202, 203, 224, 231, 235, 260, 268, 277, 293, 303, 299, 329, 331]
name5 = 'house_2 or less people_after 1965'
# house, 3 or more ppl, before 1965
home_list6 = [92, 96, 99, 133, 134, 138, 146, 152, 154, 164, 166, 168, 174, 176, 185, 187, 192, 195, 199, 200, 206, 210, 211, 222, 229, 232, 230, 243, 245, 247, 261, 264, 263, 270, 273, 290, 306, 307, 326, 327, 330]
name6 = 'house_3 or more people_before 1965'
# house, 3 or more ppl, after 1965. house 223 has no electric-mains issued
home_list7 = [66, 69, 79, 109, 110, 115, 129, 113, 170, 173, 181, 191, 207, 233, 236, 242, 246, 254, 316, 318, 325, 335]
name7 = 'house_3 or more people_after 1965'

homeList = [home_list0, home_list1, home_list2, home_list3, home_list4, home_list5, home_list6, home_list7]
names = [name0, name1, name2, name3, name4, name5, name6, name7]
for i in range(8):
    featParse = featureParser(homeList[i], names[i])
    df = featParse.homeData
    #df = df.sort_values('time')
    plotAvgwithConfInt(df, names[i])
    #df.drop(columns = ["max_val"], inplace = True)
