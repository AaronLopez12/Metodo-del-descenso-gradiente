import numpy as np
import sympy as sp
import os

x, y = sp.symbols("x y")

class Gradiente_Backtracking:
	"""
		Clase para el algoritmo descenso del gradiente
		Parámetros: 
			- Punto_inicio: un punto donde el algoritmo empieza
			- alfa_cero: tamaño de paso inicial
			- n_iteraciones: número de iteraciones 
	"""

	def __init__(self):

		os.system('clear')

		print("\n# --------------- INICIALIZACION --------------#")
		print("\nCuadrática......1")
		print("Beale...........2")
		print("Booth...........3")
		print("Himmelblau......4")

		seleccion = int(input("\nIngresa el número de la función deseada: ")) 

		alfa_cero = float(input("\nIngresa el valor de alfa cero inicial: ")) 
		
		n_iteraciones = int(input("\nIngresa el numero de iteraciones: ")) 
		
		x_inicial = float(input("\nIngresa el punto de inicio de x: ")) 
		
		y_inicial = float(input("\nIngresa el punto de inicio de y: ")) 

		self.pk = np.array([0,0])
		self.ak = alfa_cero
		self.gradiente_f = np.array([0,0])
		self.puntos = np.array([0,0])
		self.puntos_inicio = np.array([x_inicial, y_inicial])


		epsilon = 10**-5

		if seleccion == 1:
			self.f = self.Cuadratica()

		elif seleccion == 2:
			self.f = self.Beale()

		elif seleccion == 3:
			self.f = self.Booth()

		elif seleccion == 4:
			self.f = self.Himmelblau()

		else:
			print("Error en la seleccion, comience de nuevo.")
			os._exit(0)

		print("\n# --------------- INICIO ITERACIONES --------------#")


		for iteraciones in range(n_iteraciones):
			

			# Calculo de la direccion de descenso
			self.pk = self.calculo_tamaño_paso(self.puntos_inicio, self.f)
			
			if type(self.pk) is bool:
				break

			print("\n Iteracion", iteraciones, "/", n_iteraciones)

			# Backtracking a_k
			self.ak = self.Backtracking_Line_Search(self.puntos_inicio, self.ak, self.f)
			
			# if self.ak == 0:
			# 	break
			
			# Actualización de x_k+1 = x_k + pk*ak
			self.puntos = self.puntos_inicio + (self.ak*self.pk)

			f_k = self.f.subs({x: self.puntos[0] , y: self.puntos[1]})
			print("x:", self.puntos[0], "y:",self.puntos[1], "f(x,y):",f_k)

			# Verificación de la condición de paro
			paso = np.linalg.norm(self.puntos_inicio - self.puntos)
			
			if paso**2 < epsilon:
			 	break

			self.puntos_inicio = np.copy(self.puntos)

		print("\n# --------------------- FIN DE ITERACIONES -----------------#")
		print("El óptimo calculado con el método del gradiente es: ", self.puntos)

	def calculo_tamaño_paso(self, puntos, funcion):
		sbl = sp.Matrix(sp.symbols("x y"))
		f_matrix = sp.Matrix([funcion])
		f_matrix = f_matrix.jacobian(sbl).T
		gradiente_matriz = f_matrix.subs({x: puntos[0] , y: puntos[1]})
		
		self.gradiente_f[0] = gradiente_matriz[0].evalf()
		self.gradiente_f[1] = gradiente_matriz[1].evalf()
		norma = np.linalg.norm(self.gradiente_f)
		
		if norma != 0:
			return -1*self.gradiente_f / norma
		
		else:
			return False
	
	def Backtracking_Line_Search(self, pntos_k, a0, fcn):
		
		a = a0
		ro  = 0.1
		c   = 10**(-6)
		gradiente_x_pk = np.dot(self.gradiente_f.T , self.pk)

		prod = c * a * gradiente_x_pk

		while fcn.subs([(x, pntos_k[0] + a*self.pk[0]), (y, pntos_k[1] + a*self.pk[1])]).evalf() >= fcn.subs([(x, pntos_k[0]), (y,pntos_k[1])]).evalf() + prod:
			a = ro * a
		return a

	def Cuadratica(self):
		return (x - 10)**2 + (y-10)**2 
		
	def Beale(self):
		p1 = (1.5 - x + x*y)** 2
		p2 = (2.25 - x + x*(y**2))**2
		p3 = (2.625 - x + x * (y**3))**2
		return p1 + p2 + p3

	def Booth(self):
		return (x + 2*y - 7)**2 + (2*x + y - 5)**2
 	
	def Himmelblau(self):
		return (x**2 + y -11)**2 + (x+(y**2)- 7)**2



if __name__ == "__main__":
	
	Gradiente_Backtracking()
