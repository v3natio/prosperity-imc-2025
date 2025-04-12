import csv
import statistics

# Datos de usuarios y sea shells para India
india_users = [
    'Heisenberg', 'T-Radar', 'Roorkeepelago', 'Curd Rice', 'IslandPaglu',
    'Habibi', 'El Dorado', 'Zenga', 'Shibuya', 'data hawks',
    'Tradebot', 'BITS_Ogygia', 'Gamble\'s Reign', 'Pondu warriors', 'Bermuda Traders',
    'AlgoZen', 'FlyingJaat', '4_random_guys', 'Oracle28', 'London',
    'Quant_saga', 'IITBAWANA', 'devprayag', 'Arush Gupta', 'E-lemon-aters',
    'Bayesian Bay', 'black_box_crew', 'M.K. Super', 'Synchronised Mind', 'West India Comedy',
    'Pakshi', 'FloatingSapphire', 'Xteen', 'EleventhHour', 'Little boy',
    'Squad0', 'Coralspire', 'ISI Quantscape', '$Epstein', 'Edge of the World',
    'Siddadi', 'Geothermal_Trading', 'GANGDI', 'Algo Atoll', 'Fell-off',
    'StreetLoaferz', 'Sub Potro', 'eltoronto', 'MM:FarFromQuant', 'PapaBolo',
    'fsoc', 'Market Assassins', 'ringsellers', 'Sid_Dont_Lose', 'PolarVector',
    'TestCase', 'Maurya', 'ADA Cove', 'QuantX', 'Bangler',
    'OptiCoreSS', 'π\'d Piper', 'Overfitting', 'TEST', 'Elbaf',
    'Team Blue', 'TunTunMossi', 'TSP', 'jambucha', 'WaterSeven',
    'travelling Salesman', 'Dead', 'IslandAlphas', 'shresth_castle', 'Sorcerers',
    'BITS', 'Le Isla de Newmsy', 'The Hammer', 'Quantrived Isle', 'isle of math',
    'Silverstrand', 'ATLantis', 'TheLast1Q', 'beautifulworld', 'Laughing to the Bank',
    'Gosaaar', 'Loafmaxers', 'Noah\'s Ark', 'Da Pit', 'Devbhoomi',
    'Theta Drip', 'Unga_bunga_dunga', 'Straddle Isle', 'island isn\'tland', 'Too Sigma',
    'Quant Anna', 'Mathura', 'Positron', 'TakionaMuteashi', 'S^2 Capitals',
    'My_land', 'PILLALAND', 'JAMBUCHA+2', 'W16', 'Vencen',
    'Patel & Co.', 'EigenBandits', 'CrackLandia', 'Acme corp', 'ABNAS',
    'Bvin', 'Improptu paradise', 'Butterfly Effect', 'misery', 'Sindria',
    'M&M', 'Interceptor', 'Nimbus', 'Falcon', 'Ludia',
    'Cucumber', 'Stonkholm', 'Skill issue?', 'CCASHIERSS', 'Marineford',
    'IIT_Patna', 'FinBros', 'Balasamanta', 'Bull Paradise', 'MACDaddies',
    'Bikini Atoll', 'Volatility God', 'Suntracker', 'Velocity Trade', 'MarineFront',
    'Sololeveling', 'Tphy', 'Karma', 'Stonk Traders', 'Wayne',
    'Overfit', 'Isla Quantica', 'MEGA-dragonball', 'BogoIsland', 'Cherry',
    'Spacetime', 'WHIM', 'DeepThinkrs', 'Lagoon Arbitrage', 'AbundantlyRandom',
    'Lessdoit', 'Kadamba', 'ISTALAND', 'Trader.', 'Kanto',
    'blackcattraders', 'Akki\'s Island', 'Spaz17', 'JosD92', 'Wolf of Island',
    'Broke Billionaire', 'MoCo', 'Erangal', 'BlackTrader', 'Euler',
    'Kala Pani', 'Heavenly Bananas', 'Mastaane', 'KGP_Kapital', 'Bird Hunters',
    'Island KGP', 'Alibaba ki gufa', 'N/A', 'Beach Bums', 'Bounty_War',
    'Alpha Isle', 'Aeroland', 'DoPy BITS Goa', 'Wolfofbrickstreet', 'ProfitProphets',
    'epst_in', 'BreakingIsland', 'Prosperity__3', 'Quantum Quants', 'NusaPenida',
    'Bombay Bulls', 'MarketMaharajas', 'Quantum Synergy', 'HappyCat', 'shutter island',
    'Paradis Island', 'Zzzzzz', 'saddest-clown', 'KGPians', 'DepreciatingAssets',
    'LalliBigBrain', 'RAL', 'Mystery', 'Dockchip', 'CENT',
    'Port Royal', 'Powai pookies', 'Dami Dami', 'Rockers1816', 'Ghost riders',
    'Hello_12', 'mrRobot', 'ProsperLand', 'ASB', 'Underwater Holdings',
    'Beginners_Luck', 'Malibu Club', 'Xenora', 'sonder', 'ColdHarbor',
    'alphadeep', 'Auronis', 'Tradehaven', 'Jambura', 'Maverik',
    'DevinTheDuck', 'Finquark', 'Pro-vertos', 'ChaiLand', 'boombaam',
    'arbitragia', 'Zevil', 'CardinalRulers', 'Rishank', 'Salt',
    'Grand Line', 'Stauros Horcrux', 'Trident Isle', 'Greed', 'DiamondHands',
    'Ray', 'robinson_crusoe', 'epsilon@IITB', 'The CGPA Sinkhole', 'Nexa',
    'Dfe', 'Fake Analysis', 'SnackOverflow', 'Neversettle', 'Kranikan',
    'ortbo', 'Qarth', 'sigma-square', 'Canyon', 'Portfolio Point'
]

india_sea_shell_values = [
    197.422, 169.574, 109.509, 105.604, 103.741,
    100.185, 98.650, 98.108, 97.132, 97.000,
    96.693, 96.554, 96.054, 95.349, 94.240,
    93.332, 92.573, 92.374, 92.218, 92.186,
    92.143, 92.002, 91.759, 91.136, 91.129,
    90.871, 90.692, 90.621, 90.540, 90.313,
    90.313, 90.303, 90.248, 90.156, 90.077,
    89.686, 89.539, 89.538, 89.538, 89.538,
    89.537, 89.537, 89.537, 89.537, 89.537,
    89.537, 89.537, 89.537, 89.537, 89.452,
    89.350, 89.286, 89.245, 89.223, 89.172,
    89.124, 89.089, 89.057, 88.998, 88.996,
    88.991, 88.938, 88.937, 88.927, 88.903,
    88.808, 88.776, 88.743, 88.700, 88.651,
    88.645, 88.589, 88.585, 88.467, 88.390,
    88.390, 88.390, 88.390, 88.390, 88.390,
    88.390, 88.390, 88.390, 88.390, 88.382,
    88.231, 88.231, 88.185, 88.178, 88.041,
    88.039, 87.990, 87.854, 87.833, 87.833,
    87.833, 87.833, 87.833, 87.833, 87.815,
    87.807, 87.807, 87.807, 87.782, 87.681,
    87.671, 87.650, 87.618, 87.531, 87.473,
    87.461, 87.426, 87.416, 87.416, 87.416,
    87.384, 87.384, 87.270, 87.170, 87.158,
    87.038, 86.949, 86.926, 86.795, 86.782,
    86.777, 86.773, 86.682, 86.506, 86.449,
    86.393, 86.366, 86.295, 86.197, 86.174,
    86.167, 86.135, 86.133, 86.066, 85.913,
    85.519, 85.213, 85.151, 85.126, 85.082,
    84.977, 84.969, 84.847, 84.661, 84.579,
    84.476, 84.403, 84.390, 84.327, 84.307,
    84.075, 83.802, 83.791, 83.614, 83.606,
    83.515, 83.179, 82.960, 82.665, 82.449,
    82.445, 82.372, 82.354, 81.439, 81.374,
    81.367, 81.359, 80.914, 80.885, 80.860,
    80.711, 80.431, 79.548, 79.529, 79.310,
    79.293, 79.032, 78.255, 77.559, 76.765,
    76.160, 75.837, 75.669, 75.496, 75.356,
    74.773, 74.122, 74.020, 73.564, 72.798,
    72.131, 72.041, 71.567, 70.686, 70.351,
    70.187, 69.994, 69.612, 69.605, 68.556,
    68.031, 67.688, 67.183, 67.017, 66.753,
    66.625, 65.957, 65.667, 65.568, 65.541,
    64.995, 64.907, 64.715, 64.313, 64.013,
    63.896, 63.385, 63.345, 63.186, 62.966,
    62.569, 61.578, 61.502, 60.963, 60.844,
    60.689, 60.619, 60.122, 59.841, 59.089,
    59.037, 58.966, 58.941, 58.853, 58.510,
    57.968, 57.750, 57.750, 57.667, 57.631,
    57.630, 57.570, 57.392, 57.263, 56.625
]

# Datos de usuarios y sea shells para Estados Unidos
us_users = [
    'lanceball', 'Trade Haven', 'BananaDomain', 'CMU Physics', 'ivammm',
    'Freak Island', 'avengers', 'MKC Associates', 'Lehman Bros & Sis', 'MOMOLand',
    'uiuc JV team #5', 'Bink', 'A Good Island', 'Planthonys', 'UT_Wall_Street_Bets',
    'Three Kingdoms', 'hot pot', 'Charli xcx fans', 'QuantNation', 'aland',
    'Aura Farmers', 'Poverty Peninsula', 'TigerQuant Titans', 'Mallet Island', 'Thalassa Exchange',
    'galerina marginata', 'Boonstra', 'David', 'CCapital', 'trader @ maryland',
    'Antifragile', 'The Black Pearl', 'Unnamed Island', 'Crunch Rock', 'sexy island 5678',
    'spacepirate', 'Encore 1/3 Regs', 'Pink phoenix', 'Shallow Chill', 'Blue Jay Crew',
    'CUMAFN', 'CHICKEN JOCKEY', 'Hollister Island', 'Sunrun City', 'MFAMS',
    'UVA', 'RRB', 'Livermore', 'KKPP Capital', 'Math Island',
    'Meowest', 'UIUC', 'Brown Beaver', 'Henry', 'Aelegis',
    'The Data Theorists', 'Mofongos', 'Bafoon', 'nirahland', 'Borgus',
    'A^TA', 'Baby Oil', 'mu traders', 'Sweet treats', 'Alpha Animals',
    'Penn State Quants', 'QuantGPT', 'Hedging Private Ryan', 'South by Northwest', 'Lumon',
    'orange juice', 'AmethystParadise', 'Rapa Nui', 'Want what bicycle', 'SVB',
    'IMC_Trade', 'texas', 'Sinking', 'EdRuSo', 'TQD 2.0',
    'Clash', 'Devil Quant Team D', 'ArchegosCapital', 'eps-island-v2', 'punter whales',
    'Hedge Trimmers', 'Out!', 'odd-goose', 'Islanders of Plutus', 'Jai Shree Ram',
    'AYA', 'Warren Buffoons', 'All-In', 'Nassau', 'teamName',
    'Rstu', 'Princeton 2027', 'Flash Alpha', 'sunsong', 'Jewel of the Prairie',
    'BangBang', 'dumb_traders', 'Tech Savs', 'Zebedee Datsci', 'exit(255);',
    'Cloudwalker', '456', 'Island 1', 'Baoland', 'HireMePls',
    'GMT', '401 Kay', 'AlwaysLearning', 'BYOM Cove', 'WhatHappenedToGLD',
    '.', 'Los Tralalalitos', 'myf', 'KINGLUCAS', 'wait what',
    'Goated Island', 'B', 'Better than BSCF', 'insider trader', 'Shaba laba do',
    'Gooberz', 'dlake', 'poney', 'IMC Poverty', 'Team Top',
    'Curatio', 'JPCN', 'Laxmi Chit Fund', 'Z', 'alpha quants',
    'Hermitage', 'Bellerophon', 'cool island 5678', 'Columbia Island', 'team hyun',
    'tet-trio', 'Rentech', 'NU Browns', 'mafia mundeer', 'Puerto Rico',
    'T(ree) Baggers', 'Island Cuba', 'Nuggetland', 'DevilQuant A', 'Midas666',
    'TheCheese', 'We Love Arb', 'Brawlers', 'Sigma Squad', 'LEMBR',
    'XOTWOD', 'Citadel', 'Burger Flippers', 'Mathematican\'t', 'RiceOwls',
    'Mackey', 'VeryProsperous', 'Coding boats', 'Hawk Isle', '_sisyphus',
    'Little Red Book', 'FLUFFY KOALAS', 'Madhav\'s Money Maker', 'beaverparadise', 'ARMAgeddon',
    'Flotsam Island', 'Big Money', 'Gautam\'s Island', 'D1 doom scrollers', 'JustKeepProspering',
    'Gone Fishin', 'Falling Wedge', 'Jagarlamudi', 'AddyDaddy', 'Lkkkkk',
    'FliffInterns', 'pa', 'DC9', 'Cust Flow', 'LLM',
    'Abakwa', 'Pandera', 'Sunstone Isle', 'Fronk', 'Astroworld',
    'BevoWolfie1', '4 Brown, 1 Silver', 'Tropical Goats', 'Marco\'s Island', 'michael_w',
    'Beige Island', '💰🤑💰', 'Dan', 'yumpy', 'Constantinople',
    'Test-Island', 'Longhorn Liquidity', 'Hustling Hokies', 'Number go up', 'Squawker Island',
    'Himothies', 'Uright but we ALLIN!', 'BeatTheBest', 'Yale BARJ', 'pran',
    '4b1w', 'DeepAlpha', 'Conquistadors', 'Isle of Punt', 'yolo666',
    'Light Green', 'BuzzCoin', 'The Quant Ducks', 'caltech', 'Original Oasis',
    'Gamblers Grotto', 'hmmmmmmmmm', 'dm', 'Hire Me', 'Poseidon',
    'Avenged', 'BadPokerPlayers', 'Nancy\'s Notebook', 'WolTrading', 'Epsilon',
    'matchacita', 'Jerry Co.', 'h', 'z33333', '~',
    'Isle of Goat', 'Order of the Trade', 'Servette', 'DevilQuant C', 'Tierra Del Quantgo',
    'TVSCS_1', 'Famous-Chili', 'EIMERS', 'That\'s my quant', 'Fortune',
    'Prospernauts', 'PROFIT_ISLAND', 'Peekquities', 'DMV\'s Finest', 'Jain Street'
]

us_sea_shell_values = [
    111.366, 108.586, 108.560, 107.237, 103.296,
    102.843, 102.777, 102.552, 102.129, 101.903,
    100.840, 100.822, 99.961, 99.589, 99.466,
    99.203, 96.825, 96.357, 96.169, 95.937,
    95.747, 95.030, 94.921, 94.297, 94.296,
    94.126, 94.040, 93.831, 93.758, 93.167,
    92.923, 92.865, 92.719, 92.588, 92.560,
    92.538, 92.521, 92.503, 92.488, 92.392,
    92.324, 92.026, 92.008, 91.878, 91.792,
    91.566, 91.516, 91.510, 91.466, 91.405,
    91.363, 91.328, 90.902, 90.835, 90.757,
    90.722, 90.657, 90.650, 90.596, 90.506,
    90.479, 90.456, 90.386, 90.386, 90.354,
    90.313, 90.313, 90.082, 89.887, 89.878,
    89.569, 89.549, 89.538, 89.538, 89.538,
    89.538, 89.538, 89.537, 89.482, 89.449,
    89.446, 89.409, 89.360, 89.347, 89.223,
    89.223, 89.210, 89.208, 89.200, 89.120,
    89.099, 89.029, 89.012, 88.958, 88.953,
    88.879, 88.871, 88.739, 88.738, 88.532,
    88.522, 88.487, 88.487, 88.483, 88.482,
    88.468, 88.440, 88.422, 88.405, 88.390,
    88.390, 88.390, 88.390, 88.367, 88.256,
    88.233, 88.189, 88.152, 88.144, 88.065,
    87.981, 87.965, 87.960, 87.908, 87.881,
    87.872, 87.833, 87.833, 87.833, 87.833,
    87.833, 87.833, 87.833, 87.833, 87.801,
    87.781, 87.732, 87.711, 87.635, 87.633,
    87.539, 87.526, 87.493, 87.457, 87.438,
    87.409, 87.344, 87.333, 87.333, 87.328,
    87.313, 87.214, 87.193, 87.175, 87.061,
    87.053, 87.034, 86.961, 86.888, 86.870,
    86.821, 86.818, 86.810, 86.768, 86.733,
    86.726, 86.715, 86.708, 86.690, 86.671,
    86.445, 86.415, 86.338, 86.222, 86.207,
    86.125, 86.048, 86.045, 86.004, 85.950,
    85.936, 85.935, 85.882, 85.880, 85.833,
    85.801, 85.742, 85.703, 85.569, 85.568,
    85.494, 85.361, 85.280, 85.157, 85.151,
    85.148, 85.108, 85.046, 84.900, 84.845,
    84.715, 84.706, 84.699, 84.695, 84.588,
    84.566, 84.533, 84.431, 84.364, 84.306,
    84.306, 84.224, 84.217, 84.193, 84.121,
    83.951, 83.865, 83.834, 83.743, 83.733,
    83.701, 83.656, 83.621, 83.620, 83.617,
    83.485, 83.396, 83.396, 83.377, 83.292,
    83.196, 83.167, 82.978, 82.963, 82.963,
    82.910, 82.787, 82.756, 82.685, 82.309,
    82.229, 82.169, 82.134, 82.074, 81.925,
    81.852, 81.813, 81.650, 81.571, 81.312
]

# Datos de usuarios y sea shells para Estados Unidos
china_users = [
    'NoGambleNoFuture', 'ShowMeTheMoney', 'Echopeak Island', '2025_yyz', 'O.M.C.A.',
    'Armed Vehicle 2080', 'Island0xB7', 'Mistfall Isle', 'Whispering Isle', 'LALA999',
    'DEEPSICK', 'Silverhollow Isle', 'DDDOOO', '0x0', 'foxfoxGoGoGo',
    'Moonveil Atol', 'Dirty Paws', 'Koala Elysium', 'Palmrest Cove', 'Dragontato',
    'Liar\'s Poker', 'mamutde trois roi', 'chilemanin', 'YAC', 'lebron james',
    'Quantum Quant', 'JZKC', 'IMS', 'Coralshade Island', 'TremendousSaltFish',
    'Atlas___', 'Fairy Hammer', 'VP', 'CPY+ZQL', 'Letrica',
    'Mysterious', 'Shabondi', 'bupt_agent', 'alphaGrabbers', 'Flashipelago',
    '3Bro1Sis', 'goldcrest', 'Navi', 'Phoenix No.1', 'C.W.',
    'Goldenfin Atoll', 'Chicken island', '67182', 'SeaShe11 Maker', 'hahaha🔥',
    'Milkdou Island', '杭州条头糕', 'HaoBaIsland', 'JaneStreet', 'Azurebay Isle',
    'Victora', 'Sunfire Key', 'the outliers', 'DLG', 'Eleven',
    'Xufei_star_up', '2000>3000', 'quant queens', 'Z1', 'Nuonuo\'s Island',
    'Billionaires', 'Xin', 'Ave Mujica!', 'wonderland', 'mike1',
    'Sakura', 'Pulse of Alpha', 'WWWWWW', 'gggggggggggg', 'KSH',
    'QEdgeQ', 'MadIsland', 'Island-R', 'vivlavida', 'A_Basic_Bro',
    'SmileBlink', 'threeplustwo', 'LoLi', 'MinghuiLi', 'Egg Somnium Island',
    'Quaniel', 'Old Turtle', 'PigFeederKing', 'name', 'Peach Blossom',
    'Come and See See', 'killing the eve', 'gt', 'AlphaIntensivScience', 'WhisperingSnowflakes',
    'Pentagram', 'EN-', 'SG-trading', 'SmartShare', 'SDDD',
    'Cokernel', 'Midway', 'No island here', 'Emma\'s Wonderland', 'TraderZ',
    'Loong', 'Notts', 'Tan Studio', 'Unkwown', 'Friday',
    'Osmanthus', 'Long Island City', 'WinWin', 'Boba Land', 'Ilnapevoli',
    'stacknotflow', 'place_holder', 'samloker', 'Elden Souls', 'KingCrab',
    'MoneyMotivation', 'yyds', 'OnePiece3zZ', 'Arc', 'Gloria1213',
    'Kjane', 'JustRush', 'JZ', 'Spark Isle', 'BESUCCESS',
    'Fuzze Ace Island', 'Tintin Island', 'LittleDragon', 'make_money', '和我搞一辈子量化吧喵，我什么都会做的',
    'taiji', 'SKY', 'Every Trade Matters', '$LAICAI$', '66666666',
    '彳亍', 'EverBrook', 'Jolli大bee', 'verver', 'WGD',
    'G27', 'Josh\'s Island', 'First', 'Booth_layoffs', 'LLMteam',
    'Pixel Bloom', 'ConnieChen', 'duckduck', 'VIX overflow', 'Hanxiao Zhu\'s Island',
    'WWW', 'Wuye', 'The Big Bang', 'NAIVE PARK', 'Admirals',
    'EverBrook Capital', 'Billionaire', 'Boulder', 'No Sanity', 'SDSZ',
    'Sandbox_Ahoy', 'Damon Gilbert', '​Thule', 'Bertatron', 'Luckyme',
    'siyu_solo', '蓝梦-Blue Dream Co.Ltd', 'L', 'MANUALTRADER', 'AZ',
    'NullPointerException', 'China Mainland', '3832414122', 'GOAT', 'NanKai22',
    'JustOneMoment', 'Smart Money Island', 'StevenLand', 'ichigo-land', 'Brolic',
    'Island01', 'Glacier', 'NGTC', 'this is not my', 'Zeon',
    'Theshy', 'Y0816', 'QuantPanda'
]

china_sea_shell_values = [106.482, 101.166, 98.390, 97.154, 94.660, 94.457, 93.338, 93.257, 92.993, 92.993, 92.993, 92.993, 92.113, 90.947, 90.394, 89.978, 89.695, 89.695, 89.650, 89.537, 89.537, 89.537, 89.527, 89.468, 89.387, 89.385, 89.219, 88.935, 88.802, 88.744, 88.678, 88.650, 88.485, 88.390, 88.390, 88.085, 88.007, 87.879, 87.846, 87.833, 87.833, 87.833, 87.614, 87.545, 87.170, 87.096, 86.952, 86.952, 86.886, 86.491, 86.078, 86.050, 85.928, 85.640, 85.596, 85.433, 85.254, 85.105, 84.685, 84.649, 83.684, 83.569, 83.505, 83.396, 83.124, 82.670, 82.561, 82.128, 82.012, 81.595, 81.126, 81.045, 80.800, 80.208, 79.756, 79.421, 78.424, 78.317, 77.544, 76.913, 74.922, 73.118, 71.366, 71.222, 70.849, 70.842, 65.351, 63.860, 63.320, 63.208, 62.364, 62.335, 61.456, 60.327, 59.839, 59.572, 59.230, 58.589, 57.462, 56.869, 55.783, 55.765, 55.702, 55.295, 54.265, 53.349, 52.189, 51.939, 51.667, 51.667, 51.547, 51.456, 51.310, 49.468, 49.398, 49.096, 48.775, 48.735, 46.589, 46.444, 45.373, 45.099, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.340, 44.330, 44.160, 44.050, 43.998, 43.931, 43.493, 42.091, 41.934, 41.552, 40.683, 39.611, 39.611, 38.330, 38.021, 36.936, 36.936, 36.460, 36.215, 26.902, 26.902, 26.902, 26.450, 26.252, 24.446, 22.643, 19.968, 19.881, 19.115, 19.115, 13.052, 12.392, 9.194, 3.783, 2.124, 0.168, 0, 0, -0.032, -0.032, -2.210, -5.195, -9.599, -17.599, -17.599, -17.599, -17.599, -23.840, -37.810, -40.706, -43.863, -45.855, -81.641, -83.475, -131.936]

france_users = [
    "xplorers", "Musashi's Island", "Ponzi Squad", "CBC", "GoldenTurtle", 
    "LordJayS", "PUMPAJ!!!", "L&C Capital", "NoHedgeZone", "InspectElementPnL", 
    "Insiders' Island", "Panda", "Xception", "Île de Madagascar", "The French In London", 
    "Cone Forest", "Sponge Bob's Island", "Les As de la Jungle", "Johain island", "Ophir", 
    "AlphaBaguette", "import random", "île-de-France", "Zero IQ Arbitrage", "M2QF", 
    "Eldorado", "Nino's land", "A*", "Itacus", "Greenland", 
    "Mallorca", "Lehman Brothers", "Postulator", "Bermudas", "dbx", 
    "PalaiseauPlage", "Alexandro", "BaBaBif", "Infinite P&L", "Dauntless", 
    "Silver-Woman-Piano", "Lyons", "MalakasCove", "Vol et Hybrides", "Paradise Valley", 
    "brrrrrtown", "Kerviel Key", "Creta", "Gros PnL, Gros Rspct", "Capybara Island", 
    "IslandBoyz!", "IslandBoyz", "lezz_do_this", "JauneLand", "Gibraltar", 
    "JS Market Makers", "Ricius-land", "Fat Fingers", "pakupaku", "Kenny G Island", 
    "Precarity", "Croissant Capital", "La Chap", "Alambik", "Markov", 
    "Brasil'X", "ENPC", "JackSparstocks", "ALBA", "PlaceHolder", 
    "Côtes-d'Armor", "meow", "X-Men", "Babylon", "Skulk Island", 
    "ProfitPhoenix", "Island123A", "Beginner's Luck", "Blunio", "French Bond Market", 
    "SU", "Jinopsis", "CTPE Partners", "ATACA Cove", "FBD Team", 
    "Singe", "GrosMollusque", "RM Land", "LaMala", "Elie's edge", 
    "Mugi", "Here_to_help", "sytroisland", "Rounga_Diero", "Alpha Paradise", 
    "4242", "Wyvern", "Sea Shells Dealers", "Isle of Gains", "CQLJ", 
    "Caishen", "Trigo", "Sky Is The Limit", "Fortuna Isle", "Capitalism Island", 
    "solo gaming", "Cantor Capital", "Jordi", "ESSCApés", "Ellis Island", 
    "Blue_Island", "Saul Goodman"
]

france_sea_shell_values = [
    103.080, 102.950, 100.997, 99.473, 97.977, 
    97.222, 95.071, 94.307, 93.015, 92.789, 
    91.394, 91.199, 89.931, 89.549, 89.347, 
    89.271, 88.862, 88.182, 87.833, 87.645, 
    87.539, 87.507, 87.382, 87.182, 85.332, 
    84.955, 84.354, 84.318, 83.431, 83.270, 
    81.713, 80.236, 78.138, 77.611, 76.882, 
    71.379, 69.162, 65.796, 65.411, 64.907, 
    64.869, 61.436, 61.382, 59.099, 58.561, 
    58.160, 57.987, 55.972, 55.361, 55.309, 
    54.643, 54.643, 54.628, 54.265, 54.215, 
    53.092, 51.970, 49.762, 49.467, 46.106, 
    44.819, 44.402, 44.340, 44.340, 44.340, 
    44.340, 44.340, 44.340, 44.340, 44.340, 
    43.306, 43.062, 42.572, 42.376, 40.683, 
    38.081, 38.021, 38.021, 36.215, 35.911, 
    35.266, 34.819, 29.510, 29.303, 26.902, 
    26.696, 19.115, 19.115, 19.115, 19.115, 
    18.419, 11.239, 10.914, 7.538, 0.026, 
    0, 0, -0.032, -1.372, -3.656, 
    -12.893, -14.267, -17.599, -17.672, -36.451, 
    -46.530, -70.250, -75.551, -76.734, -82.072, 
    -83.475, -96.146
]

netherlands_users = [
    "Muurstraat", "camel_case", "Delft Traders", "Tristan Da Cunha", "-", 
    "Stratton Sprucemont", "Dalao island", "Fortune Reef", "Arbland", "Pompoen1 Island", 
    "Scientia Omni", "Christmas Island", "Seven - Deuce 010", "AAAAlgo", "NM$L", 
    "Koperen Ploert", "Yese", "Econometrizz", "Ad Letiek", "The Barterland", 
    "Plain or Spicy", "DaBaBanii", "Jersey", "Negative eevee", "Bazinga", 
    "Ada Refactor", "Chimpanzee Island", "The Flying Delftmen", "Capybara Capital", "Monte carlo bay", 
    "bullandbearrock", "SlowThinker", "Moni", "Demo", "Rugpull Reef", 
    "juppiiiiii", "Blue Flame", "MSc Goons", "GIT Delta 27", "Maesbury Capital", 
    "sjanschen", "Roffa", "Tokdo", "Lala land", "Tilburg Tides", 
    "TrueVoid", "Ithaka", "Pandora", "Rotterdam010", "Voorbij", 
    "Brrrrrr", "Deltaland", "Kars's Island", "manu", "total drama island", 
    "chefs' island", "Osm", "EI-LAND", "ShellCollectorsCorp", "Robert Wetzels", 
    "Shin Ramyun Box", "Hopefully not Last", "AA", "Gammesquees", "MEIJ TRADING", 
    "Risk-Free Cowboys", "NuSigma", "Polygon", "Daemon Beachgoers", "XGBoost", 
    "E-manual", "Data Miners", "premaster EUR", "CrustFund", "DonPrinter", 
    "Frame Building", "Quantillionaire", "seaaaa", "AMS-C", "dr doctor advisory", 
    "Practice", "Odvaha Trading", "Professor", "Republic of Akland", "JVDW", 
    "Bennie beunhaas", "VIXING", "Prosperity_3_Simon", "Greenfield Trading", "GIT DELTA TEAM Itô", 
    "Vamos a la playa", "808Island", "Eutopia", "STC", "Turquoise", 
    "Treasury", "TU DELFT mensah", "dr doctor holdings", "ZuidasQuant", "Max-code", 
    "MarketBreakers"
]

netherlands_sea_shell_values = [
    106.424, 105.941, 105.813, 99.380, 98.160, 
    94.594, 92.592, 90.726, 90.216, 90.016, 
    89.658, 89.447, 89.361, 88.390, 88.253, 
    88.199, 87.833, 87.817, 87.712, 87.085, 
    86.977, 86.563, 86.383, 86.259, 86.002, 
    85.577, 85.470, 82.989, 82.615, 79.561, 
    76.667, 76.303, 75.732, 74.188, 72.568, 
    71.791, 70.416, 69.636, 68.645, 68.283, 
    68.257, 64.991, 63.573, 63.539, 62.331, 
    61.898, 59.804, 59.312, 59.288, 58.748, 
    58.085, 57.392, 56.177, 56.127, 56.104, 
    54.781, 54.373, 54.291, 53.671, 52.101, 
    49.727, 49.375, 49.109, 49.082, 48.231, 
    47.512, 46.374, 46.341, 44.407, 44.340, 
    44.340, 44.340, 44.340, 44.340, 44.340, 
    44.340, 44.340, 44.340, 44.340, 44.340, 
    42.821, 42.711, 42.310, 38.021, 36.936, 
    26.383, 25.621, 19.115, 18.949, 18.859, 
    2.788, 1.072, 0.084, 0, -0.041, 
    -4.991, -10.680, -17.599, -50.680, -54.376, 
    -60.738
]

australia_users = [
    "yuh", "txg", "goober island", "TROPICAL TYGOON", "COTTONMOUTH",
    "Aegis Trading", "Ducklings island", "IMC Poverty^2", "Dynamic Survival", "Gondola",
    "Tundra-wawa", "Perthy", "requ1 r3d .-.", "Zoombiniville", "Tradeopolis",
    "Renaissance Tech", "Bando", "Quantium", "Uranus", "sthpth",
    "sinking", "403: Forbidden", "o'pen island", "retail_flow", "Eta Conversion",
    "Δ Delta", "Terrigal Trades", "rhite", "Perandus Gold Ring", "Leg Day Tomorrow",
    "Satoshi Shores", "Last Place :(", "Isle of Kermit", "Algo Apes", "team",
    "Breadland", "Passive Agressors", "Income Island", "Gal Pals", "Isolated Island",
    "liquidity_providers", "BankOrBust", "interstellar", "Liquidity Removers", "Hannah's Barbieland",
    "UNSW QuantSoc", "WWW222", "Sigma cap grp fnd", "Holiday", "NoCryingInTheCasino",
    "Sigma Technology", "Crumpeteers", "Racqueteering", "Cant think of a name", "Glob",
    "RATPOISON-DO-NOT-EAT", "a_derivative_island", "Vega Strategies", "Tempest", "Alfalfa",
    "Cockatoo", "Snobby Shores", "Teraders", "Spicy Chicken", "Jitters",
    "ANU", "5/10 super regs", "svenriksen", "DilmunMerchantsGuild", "Fortuna",
    "The Seyshells", "Brick Squadron", "themistan", "SellPastsBuyFutures", "asd",
    "Black Shells", "Wonton Masala", "Bananarama the 2nd", "paLa isla bonita", "Aisle Jimothy",
    "JJ", "Pearadise", "Arbitrage Isle", "Quantum Cove", "Mercury250",
    "Atlas Island", "Kelp Land", "JustinLan d", "ManDubu", "Jeffrey",
    "Ilovebuss1000", "zzzz", "Dehydrated Water", "Cooked123", "Up Only Season",
    "2damoon", "Aaryavarta", "snow", "FirePanda", "PFB Capital",
    "Panda_Island", "Neverland0308", "Segmentation fault", "RLMT", "island.",
    "Oopsie_Island", "TimTam", "Lebron island", "Hiiii", "athena_island",
    "Prosperville", "TeamXiaoyu@IMC", "The Utility Monster", "tarriftown", "Spin2Win",
    "Liquidity Lagoon", "CAC41", "Catnip Capital", "Albiisland", "Norena Iqbal",
    "LiTquidity", "Wild Chickens", "littlecove", "iron rod", "kirandolo",
    "bingster's paradise", "Spooky Island", "Epsilon Delta", "Thomas's team", "Anaxes",
    "HelloWorld", "Prosperity Pirates", "Path of IMC", "Facisious", "Swindle Shore",
    "TSSLand", "Tractor Lovers", "Popoff", "Archaic Archipelago", "Donkey Kong Country",
    "STAQletes", "0DTEpunters", "Acaii", "GPTtraders", "Lewlew",
    "Traders Islands", "Goro's island of d", "SpecuLuxe Isle", "Dark matter Capital", "cassian",
    "DPRK", "asdsdvwev", "low", "ChoccyBiccy", "Liquidity Enthusiast",
    "DONUT", "blank", "ProJupiter", "HBDJM Island", "ValkyrieCabal",
    "pinnacle", "BGFtothemoon", "Peen", "Kaisland", "Island of God",
    "Tidebound", "PlaceholderNameHere", "Team Big M", "0.0", "MoneyM",
    "BOQ Cadets", "Buy Low Sell Lower", "MoneyVoid", "The Bongs", "Exulta Capital",
    "The PrAUSperous", "The Money Mavericks", "ICD", "BraddlesIsland", "_",
    "someone", "Cool island", "Sean", "BuckyBoys", "Alpha Venators",
    "FastestTurtle", "bobby bob bob", "Geeza", "DN", "Silly Salmon Capital",
    "Arbitrage", "Bob Island", "Skibidi Sigmas", "Delta 1", "Insider Trading",
    "Life on Mars", "Neural Net Worth", "Emma", "fatricegik", "Colombroskis",
    "因果Island", "Winning / Everything", "Cayman Pen", "Hyperborea.", "Alfred",
    "Mersaults Paradise", "Second Place", "WolfofFlinderStreet", "Leocomon", "DoDo",
    "MalabarIsland", "Olive Oil", "Re:Prosperity", "Dbap", "Solaris",
    "Prosperty", "kl", "SIGMABOY", "How's the serenity", "ฅ^•ﻌ•^ฅ",
    "Astra", "Astray", "vvv", "Lebum", "AverageIsland",
    "Recsys Gang", "Yoj", "ashark", "Baliff", "luckyisland",
    "Tuvalu 2", "Mia's Island", "Zero Alpha", "Orderbook Oasis", "K1",
    "AAAA", "Ragebait Capital", "T1", "Akuna Capital", "reapertopia",
    "Finoson 2.0", "Koala Island", "sydtech", "Treeland", "To The Turnip Moon",
    "Tony's Island", "Cold Code", "ASD", "G", "Zinga Land"
]

australia_sea_shell_values = [
    107.292, 99.859, 96.206, 93.826, 92.875,
    92.554, 92.502, 92.186, 91.647, 91.566,
    90.656, 89.532, 89.252, 88.615, 88.605,
    88.501, 88.401, 88.390, 88.390, 88.192,
    88.078, 88.065, 88.045, 87.937, 87.931,
    87.906, 87.666, 87.658, 87.385, 87.168,
    87.157, 87.070, 86.769, 86.561, 85.901,
    84.428, 84.370, 84.306, 84.217, 84.217,
    84.217, 83.911, 83.877, 83.821, 83.804,
    83.460, 83.183, 82.528, 81.802, 81.494,
    81.131, 81.106, 80.996, 80.041, 79.628,
    79.567, 78.654, 78.019, 76.870, 76.854,
    75.100, 74.356, 74.291, 73.698, 73.002,
    71.526, 70.502, 70.489, 69.251, 69.235,
    67.794, 67.190, 66.051, 66.022, 65.781,
    63.528, 62.482, 62.474, 62.207, 62.141,
    61.695, 61.695, 61.625, 61.469, 61.166,
    60.949, 60.643, 60.387, 60.341, 60.268,
    60.092, 59.451, 59.211, 59.202, 58.637,
    58.232, 57.707, 57.513, 57.513, 57.365,
    57.364, 57.282, 57.163, 57.092, 56.963,
    56.963, 56.963, 56.630, 55.875, 55.869,
    55.814, 55.778, 55.505, 55.458, 55.376,
    55.263, 55.254, 55.249, 54.887, 54.859,
    54.279, 54.265, 54.215, 54.076, 53.576,
    53.327, 52.897, 52.715, 52.113, 51.983,
    50.644, 50.273, 49.488, 49.270, 48.979,
    47.853, 47.695, 47.517, 47.440, 47.359,
    46.821, 46.811, 46.441, 45.972, 45.797,
    45.787, 45.672, 45.626, 45.197, 44.867,
    44.538, 44.533, 44.375, 44.352, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 43.930,
    43.536, 41.279, 40.683, 40.683, 40.683,
    40.683, 40.683, 40.596, 38.422, 38.021,
    37.535, 37.145, 36.936, 36.936, 36.936,
    36.936, 36.936, 36.936, 36.936, 36.936,
    36.936, 36.936, 36.936, 36.936, 36.936,
    36.936, 36.936, 36.936, 36.936, 35.087,
    30.111, 19.115, 19.115, 18.419, 16.342,
    14.339, 12.623, 12.331, 10.911, 10.812,
    9.939, 5.632, 2.946, 44, 0,
    0, -3.656, -3.656, -7.940, -10.462,
    -15.087, -17.599, -17.599, -17.672, -20.428
]

uk_users = [
    "To the Moon", "Mom, I am trading", "CUATS", "Meth Lab", "Algopanda",
    "The Big Steppers", "0xffff", "r/PrincesStreetBets", "BleedPurple", "WarwickLand",
    "Buffet", "Sir Trades-a-Lot", "Profit Port", "Enlightenment", "placeholder",
    "TradingMinds", "JJB", "Lalaland123", "Notts Gamblers", "Market Movers",
    "Implied Surfers", "ProsperitEEE", "Imperial FinTech", "Money Central", "Eternal Trinity",
    "Tourist Resort", "Alchemist Isle", "Sellchelles", "five guys", "qAnt",
    "Slug Island", "Warwick", "Stockastic Capital", "The Domus", "The Island",
    "dough", "Dinamo", "MONEY FIRST", "Simping Island", "Money Mules",
    "Deanos Crew", "Smart Husky", "SeaShell Scallywags", "Trading Dogs", "DrillingDelta",
    "The Small Short", "jfer", "Quantopia", "Nirgendland", "poisson retribution",
    "_Redacted", "The Kennel", "Return to Monke", "The MT Isle", "The Semis",
    "QuantLand", "Potato Capital", "DU Market Making", "stuck in vim", "Optima LLM+",
    "Calcium Factory", "gang", "Nerf Yorick", "DU Quant", "Durham Dark Pool",
    "REGENT", "ShadyBlue", "gouna", "Gurkonomics", "Topology to Trading",
    "Eldia", "ABACUS", "ca$hbreezy", "MMMetricsss", "LehmanBrosRiskDep",
    "Jerry is back", "Prop-sperity", "Yuul", "Ahmeds Island", "Lions CCC",
    "Freps", "Diamond", "MRL", "Degrees of Freedom", "euwjan123",
    "password123", "TheBigBoys2", "HykanAce", "AMM", "Be ****** up by code",
    "Sardegna", "G2PLUS", "Don't know", "ingenuity", "9ZLH",
    "P Nott L", "Cheeky Trios", "D_1", "UWMFC", "Skye",
    "punggol", "Estia", "Yorkie", "Lore Trading", "Pikachu Isle",
    "SilenceOfTheLambdas", "JeTrade", "Rajab & Mike", "Imperial Isle", "Diddlers",
    "No Man", "DTC", "Git island", "Brecon", "sandband",
    "Tariffic Traders", "Energy Sector", "DhyeyTesting", "Garcus Man", "South Pokerbay",
    "GenericGreekGodFund", "ab2379", "🍇", "Q", "TestIsland",
    "BATHBOMB", "Alphathetical", "Aither", "merle", "Tahir Island",
    "ChChannel Islands", "Ashbeck & Buxton", "MarketJamakers", "Archipelitrage", "Warwick Swans",
    "BuyTheDip", "StonksAgain", "Roxu23", "203 Solutions", "A_R",
    "Place Holder Island", "Jeet Street", "SALMAANSOG", "Gamma", "Kainth",
    "AYVTrading", "Yega's Island", "Magpie Island", "CultureCraft", "Azura",
    "Lboro Tracy Island", "Kaista's Kingdom", "KXR", "Champ1", "Trader's Paradise",
    "Matt's Practice", "Serenity Island", "Horizontal", "Dunder Mifflin", "number 1 potentially",
    "KH", "AlgoWizard", "The_Island9", "Trade for Fun", "Durham Darkpool",
    "Wrangthorn Island", "The Sharpe Minds", "Go Big or Go Home", "Mariejois", "Incline",
    "luckyfew", "Bedfordiensis", "Sumqayit", "average", "Bullish Confidence",
    "hellothere", "Cam_123", "gigaturbo capital", "Number1 Trader", "island of morrington",
    "Lonewolf", "Vicious Volatility", "RUN", "xtung", "Pelegosto",
    "Lorentz of Arabia", "Fejø", "Mars", "Retail vs SMC", "GlebeOne",
    "Vic's island", "Sel4", "4ce of Log", "LiquidIsland", "Alameda Research 2.0",
    "Pi-rates Cove", "shetland island", "SR-71 island", "AlphaLand", "Evland",
    "Industro", "Tropic Like It's Hot", "island_0", "Fellerz"
]

uk_sea_shell_values = [
    104.236, 99.347, 98.559, 97.723, 92.887,
    92.014, 91.136, 91.097, 90.993, 90.832,
    90.278, 89.999, 89.817, 89.537, 89.465,
    89.393, 89.391, 89.256, 89.062, 88.974,
    88.676, 88.648, 88.457, 88.390, 88.237,
    88.165, 87.877, 87.833, 87.833, 87.614,
    87.471, 86.969, 86.538, 86.525, 86.345,
    86.341, 86.109, 85.912, 85.901, 85.734,
    85.636, 85.549, 85.198, 85.185, 84.982,
    84.940, 84.403, 84.157, 83.894, 83.766,
    83.197, 82.539, 81.868, 81.317, 78.109,
    77.435, 76.419, 76.419, 76.130, 74.478,
    73.607, 70.538, 70.493, 69.737, 69.521,
    67.992, 66.866, 66.551, 66.357, 65.115,
    64.784, 63.733, 62.608, 62.088, 62.007,
    61.695, 60.691, 60.354, 59.950, 59.374,
    59.013, 57.623, 57.512, 57.337, 56.817,
    56.492, 56.321, 56.010, 55.994, 55.923,
    55.834, 55.322, 55.254, 54.997, 54.635,
    54.327, 54.215, 54.116, 54.016, 53.637,
    53.460, 53.019, 53.014, 52.924, 52.480,
    51.556, 49.822, 48.161, 47.400, 47.372,
    46.138, 45.368, 45.118, 44.440, 44.401,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    43.493, 43.382, 43.340, 43.319, 42.419,
    41.795, 40.683, 40.683, 39.611, 39.526,
    38.021, 37.980, 36.936, 36.936, 36.936,
    36.936, 36.528, 36.417, 36.215, 36.215,
    35.503, 30.098, 26.902, 23.246, 21.923,
    21.176, 19.581, 19.115, 19.115, 19.115,
    19.115, 17.115, 14.342, 13.931, 11.417,
    9.431, 8.788, 6.842, 6.444, 6.444,
    5.959, 5.945, -632, -3.656, -15.283,
    -16.524, -17.599, -17.599, -17.599, -17.599,
    -17.672, -25.955, -39.983, -41.795, -51.264,
    -58.815, -60.552, -64.309, -65.695, -77.164,
    -82.709, -99.554, -106.213, -126.029
]

hk_users = [
    "Gorilla Gamma", "Ding Crab", "Granite Flow", "Five Piece is REAL", "DeepSick",
    "Vestrahorn", "H&K AlphaStat", "SU canteen", "SoloLevel", "Sicily",
    "Wolves of Hong Kong", "Elsa", "sheeeesh kebab", "Strawberry Elephant", "IMChineseUniversity",
    "SigmaCapital", "Goodisland", "HK Island", "JohnnyTung", "Y0ungM1racle",
    "Harry See", "下山虎大战飞天龙", "Yeahhhhhhhh", "Hello Island", "YYLand",
    "kek", "Casen", "Market Wizard", "Pragmatix", "Neverland",
    "JasonLam", "Bress", "Lamma Island", "CUHK Team A", "XVII",
    "leo_hkg", "Dogeffett", "certain_of_uncertain", "JC", "Snowman",
    "lolol", "Nrus", "notsea island", "Concept Crafters", "Nomads",
    "Lantau Island", "Itsztar", "Theone", "YL Island", "Arian",
    "KK_V", "Hyson's Island", "Hong Kong", "The Quantifiers", "PiratesHongKong",
    "Trying my best", "Dino", "Pluto", "luckiland", "The Lung"
]

hk_sea_shell_values = [
    102.396, 95.378, 93.233, 93.027, 92.113,
    91.695, 90.664, 89.415, 88.873, 87.833,
    86.736, 86.441, 85.940, 85.438, 83.374,
    80.475, 74.168, 68.614, 66.048, 65.187,
    61.695, 59.392, 59.376, 57.553, 56.785,
    56.226, 54.227, 53.254, 50.604, 47.527,
    44.553, 44.340, 44.340, 44.340, 44.340,
    44.340, 44.340, 44.340, 44.340, 44.340,
    44.340, 43.989, 38.994, 38.021, 36.936,
    36.936, 36.936, 18.419, 11.571, 6.444,
    2.160, 0, -240, -1.039, -17.599,
    -23.200, -23.200, -24.800, -41.937, -42.719
]

# Crear CSV para India
with open('india_sea_shells_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Rank', 'User Name', 'Sea Shells'])
    for i in range(250):
        writer.writerow(['India', i+1, india_users[i], india_sea_shell_values[i]])

# Calcular el promedio para India
india_average = statistics.mean(india_sea_shell_values)

# Crear CSV para Estados Unidos
with open('us_sea_shells_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Rank', 'User Name', 'Sea Shells'])
    for i in range(250):
        writer.writerow(['USA', i+1, us_users[i], us_sea_shell_values[i]])

# Calcular el promedio para Estados Unidos
us_average = statistics.mean(us_sea_shell_values)

# Crear CSV para China
with open('china_sea_shells_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Rank', 'User Name', 'Sea Shells'])
    for i in range(193):
        writer.writerow(['China', i+1, china_users[i], china_sea_shell_values[i]])

# Calcular el promedio para China
china_average = statistics.mean(china_sea_shell_values)

# Crear CSV para France
with open('france_sea_shells_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Rank', 'User Name', 'Sea Shells'])
    for i in range(112):
        writer.writerow(['France', i+1, france_users[i], france_sea_shell_values[i]])

# Calcular el promedio para France
france_average = statistics.mean(france_sea_shell_values)

# Crear CSV para Netherlands
with open('netherlands_sea_shells_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Rank', 'User Name', 'Sea Shells'])
    for i in range(101):
        writer.writerow(['Netherlands', i+1, netherlands_users[i], netherlands_sea_shell_values[i]])

# Calcular el promedio para Netherlands
netherlands_average = statistics.mean(netherlands_sea_shell_values)

# Crear CSV para Australia
with open('australia_sea_shells_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Rank', 'User Name', 'Sea Shells'])
    for i in range(170):
        writer.writerow(['Australia', i+1, australia_users[i], australia_sea_shell_values[i]])

# Calcular el promedio para Australia
australia_average = statistics.mean(australia_sea_shell_values)

# Crear CSV para UK
with open('uk_sea_shells_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Rank', 'User Name', 'Sea Shells'])
    for i in range(204):
        writer.writerow(['UK', i+1, uk_users[i], uk_sea_shell_values[i]])

# Calcular el promedio para UK
uk_average = statistics.mean(india_sea_shell_values)

# Crear CSV para HK
with open('hk_sea_shells_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Rank', 'User Name', 'Sea Shells'])
    for i in range(60):
        writer.writerow(['HK', i+1, hk_users[i], hk_sea_shell_values[i]])

# Calcular el promedio para HK
hk_average = statistics.mean(hk_sea_shell_values)

# Datos para el resumen total
countries_data = [
    {'Country': 'India', 'Average': india_average, 'Count': len(india_users)},
    {'Country': 'USA', 'Average': us_average, 'Count': len(us_users)},
    {'Country': 'China', 'Average': china_average, 'Count': len(china_users)},
    {'Country': 'France', 'Average': france_average, 'Count': len(france_users)},
    {'Country': 'Netherlands', 'Average': netherlands_average, 'Count': len(netherlands_users)},
    {'Country': 'Australia', 'Average': australia_average, 'Count': len(australia_users)},
    {'Country': 'UK', 'Average': uk_average, 'Count': len(uk_users)},
    {'Country': 'HK', 'Average': hk_average, 'Count': len(hk_users)}
]

# Calcular el promedio global
all_values = india_sea_shell_values + us_sea_shell_values + china_sea_shell_values + france_sea_shell_values + netherlands_sea_shell_values + australia_sea_shell_values + uk_sea_shell_values + hk_sea_shell_values
global_average = statistics.mean(all_values)

# Crear archivo de resumen total
with open('sea_shells_summary_total.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Average Sea Shells', 'Number of Users'])
    for country_data in countries_data:
        writer.writerow([country_data['Country'], 
                        round(country_data['Average'], 3), 
                        country_data['Count']])
    writer.writerow(['Global', round(global_average, 3), len(all_values)])

print(f"Archivos CSV creados para India con datos de {len(india_users)} usuarios")
print(f"Archivos CSV creados para Estados Unidos con datos de {len(us_users)} usuarios")
print(f"Promedio de Sea Shells para India: {india_average:.3f}")
print(f"Promedio de Sea Shells para Estados Unidos: {us_average:.3f}")
print(f"Promedio de Sea Shells para China: {china_average:.3f}")
print(f"Promedio de Sea Shells para France: {france_average:.3f}")
print(f"Promedio de Sea Shells para Netherlands: {netherlands_average:.3f}")
print(f"Promedio de Sea Shells para Australia: {australia_average:.3f}")
print(f"Promedio de Sea Shells para UK: {uk_average:.3f}")
print(f"Promedio de Sea Shells para HK: {hk_average:.3f}")
print(f"Promedio global de Sea Shells: {global_average:.3f}")