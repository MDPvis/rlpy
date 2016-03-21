"""
Mocked data for testing purposes.
"""

mock_target_rollouts = {
  "simplistic": [
    [{'action': 0, 'x': -0.5},
     {'action': 0, 'x': -0.5}],
    [{'action': 0, 'x': -0.5},
     {'action': 0, 'x': -0.5}]
  ],
  "simplistic opposite action": [
    [{'action': 1, 'x': -0.5},
     {'action': 1, 'x': -0.5}],
    [{'action': 1, 'x': -0.5},
     {'action': 1, 'x': -0.5}]
  ],
  "simplistic opposite state": [
    [{'action': 0, 'x': 0.5},
     {'action': 0, 'x': 0.5}],
    [{'action': 0, 'x': 0.5},
     {'action': 0, 'x': 0.5}]
    ],
  "viewport 1.0": [
    [{'action': 0, 'x': -0.5},
     {'action': 0, 'x': -0.5}],
    [{'action': 1, 'x': 0.5},
     {'action': 1, 'x': 0.5}]
  ],
  "viewport 1.0 extreme action and state": [
    [{'action': 1, 'x': 0.5},
     {'action': 1, 'x': 0.5}],
    [{'action': 1, 'x': 0.5},
     {'action': 1, 'x': 0.5}]
  ],
  "high variance state and action": [
      [{'action': 1, 'high': -10000., 'low': 10000.},
       {'action': 1, 'high': 99999999., 'low': 45.}],
      [{'action': 1, 'high': 99999999., 'low': -10000.},
       {'action': 1, 'high': 99999999., 'low': 99999999.}],
      [{'action': 1, 'high': 99999999., 'low': -10000.},
       {'action': 1, 'high': 99999999., 'low': 9999.}]
  ],
  "high variance state and action synthetic": [
      [{'action': 1, 'high': -10000., 'low': -10000.},
       {'action': 1, 'high': -10000., 'low': 99999999.}],
      [{'action': 1, 'high': -10000., 'low': 99999999.},
       {'action': 1, 'high': -10000., 'low': 99999999.}],
      [{'action': 1, 'high': -10000., 'low': 99999999.},
       {'action': 1, 'high': 99999999., 'low': 99999999.}]
  ],
  "normalization": [
    [
        {'action':1, 'x1': 100, 'x2': -1},
        {'action':0, 'x1': 0, 'x2': -1}],
       [{'action':1, 'x1': 0, 'x2': -1},
        {'action':0, 'x1': 100, 'x2': -1}]
  ],
  "MountainCarTarget": [[{'action': 0, 'x': -0.5}, {'action': 1, 'x': -0.50119211204430147, 'reward': -1, 'xdot': -0.0011921120443014763}], [{'action': 0, 'x': -0.5}, {'action': 1, 'x': -0.50118932556191675, 'reward': -1, 'xdot': -0.0011893255619167189}], [{'action': 1, 'x': -0.5}, {'action': 2, 'x': -0.50008411045206902, 'reward': -1, 'xdot': -8.4110452069051402e-05}], [{'action': 0, 'x': -0.5}, {'action': 2, 'x': -0.50111849799655273, 'reward': -1, 'xdot': -0.0011184979965527244}], [{'action': 0, 'x': -0.5}, {'action': 0, 'x': -0.50116323409195052, 'reward': -1, 'xdot': -0.0011632340919504709}], [{'action': 0, 'x': -0.5}, {'action': 2, 'x': -0.50126263579252972, 'reward': -1, 'xdot': -0.0012626357925296799}], [{'action': 1, 'x': -0.5}, {'action': 2, 'x': -0.50027279932468116, 'reward': -1, 'xdot': -0.00027279932468119214}], [{'action': 2, 'x': -0.5}, {'action': 0, 'x': -0.499121211653979270087878834602071287}], [{'action': 1, 'x': -0.5}, {'action': 1, 'x': -0.50008111933572275, 'reward': -1, 'xdot': -8.1119335722704459e-05}], [{'action': 1, 'x': -0.5}, {'action': 1, 'x': -0.50018454713171867, 'reward': -1, 'xdot': -0.0001845471317186709}], [{'action': 0, 'x': -0.5}, {'action': 1, 'x': -0.50125318811899544, 'reward': -1, 'xdot': -0.0012531881189954707}], [{'action': 0, 'x': -0.5}, {'action': 0, 'x': -0.50124817234668739, 'reward': -1, 'xdot': -0.0012481723466874479}], [{'action': 1, 'x': -0.5}, {'action': 2, 'x': -0.50017247333981929, 'reward': -1, 'xdot': -0.00017247333981924292}], [{'action': 0, 'x': -0.5}, {'action': 2, 'x': -0.5012239318817483, 'reward': -1, 'xdot': -0.0012239318817483318}], [{'action': 0, 'x': -0.5}, {'action': 1, 'x': -0.5011856129377259, 'reward': -1, 'xdot': -0.0011856129377259476}], [{'action': 1, 'x': -0.5}, {'action': 2, 'x': -0.50027308504408197, 'reward': -1, 'xdot': -0.00027308504408198622}], [{'action': 0, 'x': -0.5}, {'action': 1, 'x': -0.50115442385962472, 'reward': -1, 'xdot': -0.001154423859624773}], [{'action': 1, 'x': -0.5}, {'action': 1, 'x': -0.50008809338846638, 'reward': -1, 'xdot': -8.8093388466332424e-05}], [{'action': 0, 'x': -0.5}, {'action': 2, 'x': -0.50120494142405447, 'reward': -1, 'xdot': -0.0012049414240545}], [{'action': 0, 'x': -0.5}, {'action': 2, 'x': -0.50113731676498385, 'reward': -1, 'xdot': -0.0011373167649838044}], [{'action': 2, 'x': -0.5}, {'action': 0, 'x': -0.499143489661080130085651033891987632}], [{'action': 2, 'x': -0.5}, {'action': 0, 'x': -0.499234766491954470076523350804551097}], [{'action': 0, 'x': -0.5}, {'action': 0, 'x': -0.50121375733398443, 'reward': -1, 'xdot': -0.0012137573339844206}], [{'action': 1, 'x': -0.5}, {'action': 1, 'x': -0.5001628036500857, 'reward': -1, 'xdot': -0.00016280365008568135}], [{'action': 2, 'x': -0.5}, {'action': 0, 'x': -0.499079168236557400920831763442588}], [{'action': 0, 'x': -0.5}, {'action': 1, 'x': -0.50123506765295034, 'reward': -1, 'xdot': -0.0012350676529502903}], [{'action': 0, 'x': -0.5}, {'action': 1, 'x': -0.50114622133907616, 'reward': -1, 'xdot': -0.0011462213390761775}], [{'action': 2, 'x': -0.5}, {'action': 2, 'x': -0.499183580849598020081641915040200409}], [{'action': 0, 'x': -0.5}, {'action': 1, 'x': -0.50124504908744016, 'reward': -1, 'xdot': -0.0012450490874401535}], [{'action': 1, 'x': -0.5}, {'action': 1, 'x': -0.50014557708627616, 'reward': -1, 'xdot': -0.00014557708627620258}], [{'action': 1, 'x': -0.5}, {'action': 2, 'x': -0.50023752653183329, 'reward': -1, 'xdot': -0.00023752653183324658}], [{'action': 2, 'x': -0.5}, {'action': 2, 'x': -0.499112644358199690088735564180032972}], [{'action': 0, 'x': -0.5}, {'action': 2, 'x': -0.50110925402266948, 'reward': -1, 'xdot': -0.0011092540226694965}], [{'action': 1, 'x': -0.5}, {'action': 0, 'x': -0.50008155111116659, 'reward': -1, 'xdot': -8.1551111166578105e-05}], [{'action': 1, 'x': -0.5}, {'action': 2, 'x': -0.50008149078653119, 'reward': -1, 'xdot': -8.1490786531189836e-05}], [{'action': 0, 'x': -0.5}, {'action': 0, 'x': -0.50112899028828961, 'reward': -1, 'xdot': -0.0011289902882895969}], [{'action': 2, 'x': -0.5}, {'action': 0, 'x': -0.499220281611653990077971838834602471}], [{'action': 0, 'x': -0.5}, {'action': 0, 'x': -0.50121761496466488, 'reward': -1, 'xdot': -0.0012176149646648284}], [{'action': 0, 'x': -0.5}, {'action': 0, 'x': -0.50121324636829045, 'reward': -1, 'xdot': -0.001213246368290462}], [{'action': 0, 'x': -0.5}, {'action': 2, 'x': -0.50126401350489946, 'reward': -1, 'xdot': -0.0012640135048995005}], [{'action': 0, 'x': -0.5}, {'action': 2, 'x': -0.50116352271332798, 'reward': -1, 'xdot': -0.0011635227133279422}], [{'action': 1, 'x': -0.5}, {'action': 1, 'x': -0.50017219339347596, 'reward': -1, 'xdot': -0.00017219339347591733}], [{'action': 1, 'x': -0.5}, {'action': 0, 'x': -0.50016165370505805, 'reward': -1, 'xdot': -0.0001616537050580214}], [{'action': 1, 'x': -0.5}, {'action': 1, 'x': -0.500213129213679, 'reward': -1, 'xdot': -0.00021312921367899252}], [{'action': 1, 'x': -0.5}, {'action': 0, 'x': -0.50025048343168843, 'reward': -1, 'xdot': -0.00025048343168837886}], [{'action': 1, 'x': -0.5}, {'action': 2, 'x': -0.50021896178557979, 'reward': -1, 'xdot': -0.00021896178557981706}], [{'action': 0, 'x': -0.5}, {'action': 1, 'x': -0.50115954041720723, 'reward': -1, 'xdot': -0.0011595404172072407}], [{'action': 2, 'x': -0.5}, {'action': 0, 'x': -0.499111054998325810088894500167421538}], [{'action': 2, 'x': -0.5}, {'action': 0, 'x': -0.499141279696809990085872030318998884}], [{'action': 1, 'x': -0.5}, {'action': 2, 'x': -0.50012980419974473, 'reward': -1, 'xdot': -0.00012980419974473829}]]
}

