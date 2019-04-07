import numpy as np

"""
Base class for element
@ Sylvain DUBREUIL, ONERA
"""

class finite_element():
    """
    Construction:\n
    Inputs:\n
    element_name: string
    """
    def __init__(self,element_name, element_property, material):
        self.element_name = element_name
        self.element_property = element_property
        self.material = material
    """
    Function that compute the elementary rigidity matrix in the global coordinate system
    """
    def compute_K(self,nodes):
        if self.element_name == "Euler" or self.element_name == "Timoshenko":
            from beam_elements import beam_element
            element = beam_element(self.element_property,self.material,self.element_name)
            K_elem = element.compute_K_elem(nodes)
        if self.element_name == "DKT":
            from tri_elements import DKT_element
            element = DKT_element(self.element_property, self.material)
            K_elem = element.compute_K_elem(nodes)
        if self.element_name == "Q4":
            from quad_elements import Q4_element
            element = Q4_element(self.element_property,self.material)
            K_elem = element.compute_K_elem(nodes)
        return K_elem   
    """
    Function that compute the stress and strains in the global coordinate system
    """
    def compute_strain_and_stress(self,nodes_coord,U_elm):
        from tri_elements import DKT_element
        element = DKT_element(self.element_property, self.material)
        strain,stress = element.compute_strain_and_stress(nodes_coord,U_elm)
        return strain, stress