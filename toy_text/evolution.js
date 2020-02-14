document.getElementById('game')
level= [['S','F','F','F'],['F','H','F','H'],['F','F','F','H'],['H','F','F',G]]

for i in range(4):
	for j in range(4):
		span=document.createElement('span')
		span.innerHTML=level[i][j]
		game.appendChild(span)