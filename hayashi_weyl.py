# hayashi_weyl_general.py
"""
Hayashi-style Quantized Weyl Algebra in n variables.

Defining relations (for each i, j):
    g_i * g_i^{-1} = 1 = g_i^{-1} * g_i
    g_i * g_j = g_j * g_i
    g_i * x_i = q * x_i * g_i
    g_i * dx_i = q^{-1} * dx_i * g_i
    g_i * x_j = x_j * g_i          (i != j)
    g_i * dx_j = dx_j * g_i        (i != j)
    x_i * x_j = x_j * x_i
    dx_i * dx_j = dx_j * dx_i
    dx_i * x_j = x_j * dx_i        (i != j)
    dx_i * x_i = q * x_i * dx_i + g_i^{-1}

From these relations one derives:

    (1) x_i * dx_i = (g_i - g_i^{-1}) / (q - q^{-1})
    (2) dx_i * x_i = (qg_i - (qg_i)^{-1}) / (q - q^{-1})

'PBW' basis: 
    x1^{a1} * ... * xn^{an} * dx1^{b1} * ... * dxn^{bn} * g1^{e1} * ... * gn^{en}
Using (1) and (2), we assume the numbers a1, ..., an, b1, ..., bn, satisfy aj * bj = 0.
"""

from sympy import Symbol, expand, simplify, latex, S, together
from collections import defaultdict


class HayashiAlgebra:
    """
    Factory class for Hayashi quantized Weyl algebra in n variables.
    """
    
    def __init__(self, n, x_names=None, dx_names=None, g_names=None):
        """
        Initialize algebra with n pairs of variables.
        
        Args:
            n: number of variable pairs (x_i, dx_i, g_i)
            x_names: list of names for x variables, e.g. ['x1', 'x2', 'y']
            dx_names: list of names for dx variables, e.g. ['dx1', 'dx2', 'dy']
            g_names: list of names for g variables, e.g. ['g1', 'g2', 'w']
        """
        self.n = n
        self.q = Symbol('q', commutative=True)
        
        # Set up names (default to x1, x2, ... if not provided)
        if x_names is None:
            x_names = [f"x{i+1}" for i in range(n)]
        if dx_names is None:
            dx_names = [f"dx{i+1}" for i in range(n)]
        if g_names is None:
            g_names = [f"g{i+1}" for i in range(n)]
        
        if len(x_names) != n or len(dx_names) != n or len(g_names) != n:
            raise ValueError(f"Name lists must have length {n}")
        
        self.x_names = x_names
        self.dx_names = dx_names
        self.g_names = g_names
        
        # Create generators as HayashiElement objects
        # Indexing will be 1-based for mathematical convenience
        self.x = {}
        self.dx = {}
        self.g = {}
        self.ginv = {}
        
        for i in range(1, n + 1):
            # x_i: exponent 1 in position (i-1)
            a = [0] * n
            a[i-1] = 1
            self.x[i] = HayashiElement(self, {tuple(a + [0]*n + [0]*n): S(1)})
            
            # dx_i: exponent 1 in position n + (i-1)
            b = [0] * n
            b[i-1] = 1
            self.dx[i] = HayashiElement(self, {tuple([0]*n + b + [0]*n): S(1)})
            
            # g_i: exponent 1 in position 2n + (i-1)
            e = [0] * n
            e[i-1] = 1
            self.g[i] = HayashiElement(self, {tuple([0]*n + [0]*n + e): S(1)})
            
            # g_i^{-1}: exponent -1 in position 2n + (i-1)
            einv = [0] * n
            einv[i-1] = -1
            self.ginv[i] = HayashiElement(self, {tuple([0]*n + [0]*n + einv): S(1)})
        
        # Identity element
        self.one = HayashiElement(self, {tuple([0] * (3*n)): S(1)})
    
    def comm(self, A, B, reduce=True):
        """
        Commutator [A, B] = A*B - B*A.
    
        Args:
            A, B: HayashiElement objects
            reduce: if True (default), apply full_reduce to get true PBW basis
        """
        result = A * B - B * A
        return self.full_reduce(result).simplify() if reduce else result.simplify()

    def anticomm(self, A, B, reduce=True):
        """
        Anticommutator {A, B} = A*B + B*A.
    
        Args:
            A, B: HayashiElement objects
            reduce: if True (default), apply full_reduce to get true PBW basis
        """
        result = A * B + B * A
        return self.full_reduce(result).simplify() if reduce else result.simplify()
    
    def monomial(self, a=None, b=None, e=None, coef=1):
        """
        Create a monomial: coef * x1^{a[0]} * ... * xn^{a[n-1]} * dx1^{b[0]} * ... * g1^{e[0]} * ...
        
        Args:
            a: list of x exponents (length n), default all zeros
            b: list of dx exponents (length n), default all zeros
            e: list of g exponents (length n), default all zeros
            coef: coefficient
        """
        if a is None:
            a = [0] * self.n
        if b is None:
            b = [0] * self.n
        if e is None:
            e = [0] * self.n
        
        if any(ai < 0 for ai in a) or any(bi < 0 for bi in b):
            raise ValueError("x and dx exponents must be non-negative")
        
        mon = tuple(list(a) + list(b) + list(e))
        return HayashiElement(self, {mon: coef})
    
    def full_reduce(self, elt):
        """
        Fully reduce an element to true PBW basis.
        
        Applies x_i * dx_i = (g_i - g_i^{-1}) / (q - q^{-1}) for all i.
        """
        result = HayashiElement(self, {})
        
        for mon, coef in elt.terms.items():
            reduced = self._reduce_monomial(mon, coef)
            result = result + reduced
        
        return result
    
    def _reduce_monomial(self, mon, coef):
        """Reduce a single monomial."""
        n = self.n
        a = list(mon[:n])
        b = list(mon[n:2*n])
        e = list(mon[2*n:])
        
        # Check each index for reduction
        for i in range(n):
            if a[i] >= 1 and b[i] >= 1:
                prefactor = 1 / (self.q - self.q**(-1))
                
                # Term with g_i
                a_new = a.copy()
                b_new = b.copy()
                e_new1 = e.copy()
                a_new[i] -= 1
                b_new[i] -= 1
                e_new1[i] += 1
                coef1 = expand(coef * prefactor * self.q**(-(b[i]-1)))
                mon1 = tuple(a_new + b_new + e_new1)
                term1 = self._reduce_monomial(mon1, coef1)
                
                # Term with g_i^{-1}
                e_new2 = e.copy()
                e_new2[i] -= 1
                coef2 = expand(-coef * prefactor * self.q**(b[i]-1))
                mon2 = tuple(a_new + b_new + e_new2)
                term2 = self._reduce_monomial(mon2, coef2)
                
                return term1 + term2
        
        # No reduction needed
        return HayashiElement(self, {mon: coef})
    
    def pretty(self, elt):
        """Simplify coefficients for nicer display."""
        return elt.simplify()


class HayashiElement:
    """
    An element of the Hayashi quantized Weyl algebra.
    
    Monomials are tuples of length 3n:
        (a_1, ..., a_n, b_1, ..., b_n, e_1, ..., e_n)
    representing x1^{a1} * ... * xn^{an} * dx1^{b1} * ... * dxn^{bn} * g1^{e1} * ... * gn^{en}
    """
    
    def __init__(self, algebra, terms=None):
        self.algebra = algebra
        self.n = algebra.n
        self.q = algebra.q
        
        if terms is None:
            self.terms = {}
        else:
            self.terms = {}
            for mon, coef in terms.items():
                c = expand(coef)
                if c != 0:
                    self.terms[mon] = c
    
    def copy(self):
        return HayashiElement(self.algebra, dict(self.terms))
    
    def simplify(self):
        """Return a new element with simplified coefficients."""
        return HayashiElement(
            self.algebra,
            {m: together(simplify(c)) for m, c in self.terms.items()}
        )
    
    @property
    def is_zero(self):
        return len(self.terms) == 0
    
    @property
    def is_monomial(self):
        return len(self.terms) <= 1
    
    def coefficient(self, mon):
        """Get coefficient of a specific monomial."""
        if isinstance(mon, tuple):
            return self.terms.get(mon, S(0))
        elif isinstance(mon, HayashiElement) and mon.is_monomial:
            if mon.terms:
                key = list(mon.terms.keys())[0]
                return self.terms.get(key, S(0))
        return S(0)
    
    def __eq__(self, other):
        if isinstance(other, HayashiElement):
            if other.algebra is not self.algebra:
                return False
            # Compare simplified forms for robust equality
            self_simp = {m: simplify(c) for m, c in self.terms.items() if simplify(c) != 0}
            other_simp = {m: simplify(c) for m, c in other.terms.items() if simplify(c) != 0}
            return self_simp == other_simp
        if other == 0:
            return self.is_zero
        return NotImplemented
    
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result
    
    def __add__(self, other):
        if isinstance(other, HayashiElement):
            if other.algebra is not self.algebra:
                raise ValueError("Cannot add elements from different algebras")
        elif isinstance(other, (int, float)):
            zero_mon = tuple([0] * (3 * self.n))
            other = HayashiElement(self.algebra, {zero_mon: S(other)})
        elif hasattr(other, 'is_number') or hasattr(other, 'is_Symbol'):
            # SymPy expression
            zero_mon = tuple([0] * (3 * self.n))
            other = HayashiElement(self.algebra, {zero_mon: other})
        else:
            return NotImplemented
        
        result = dict(self.terms)
        for mon, coef in other.terms.items():
            if mon in result:
                new_coef = expand(result[mon] + coef)
                if new_coef == 0:
                    del result[mon]
                else:
                    result[mon] = new_coef
            else:
                result[mon] = coef
        return HayashiElement(self.algebra, result)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __neg__(self):
        return HayashiElement(self.algebra, {mon: -coef for mon, coef in self.terms.items()})
    
    def __rmul__(self, scalar):
        if isinstance(scalar, HayashiElement):
            return scalar * self
        return HayashiElement(self.algebra, {mon: expand(scalar * coef) 
                                             for mon, coef in self.terms.items()})
    
    def __mul__(self, other):
        if not isinstance(other, HayashiElement):
            return HayashiElement(self.algebra, {mon: expand(coef * other) 
                                                 for mon, coef in self.terms.items()})
        
        if other.algebra is not self.algebra:
            raise ValueError("Cannot multiply elements from different algebras")
        
        result = HayashiElement(self.algebra, {})
        for mon1, coef1 in self.terms.items():
            for mon2, coef2 in other.terms.items():
                product = self._multiply_monomials(mon1, mon2, expand(coef1 * coef2))
                result = result + product
        return result
    
    def __truediv__(self, scalar):
        if isinstance(scalar, HayashiElement):
            raise TypeError("Cannot divide by a HayashiElement")
        return HayashiElement(self.algebra, {mon: expand(coef / scalar) 
                                             for mon, coef in self.terms.items()})
    
    def __pow__(self, p):
        if not isinstance(p, int) or p < 0:
            raise ValueError("Power must be a non-negative integer")
        if p == 0:
            return self.algebra.one
        result = self.copy()
        for _ in range(p - 1):
            result = result * self
        return result
    
    def _multiply_monomials(self, mon1, mon2, coef):
        """Multiply two monomials, return in PBW form."""
        n = self.n
        a1 = list(mon1[:n])
        b1 = list(mon1[n:2*n])
        e1 = list(mon1[2*n:])
        
        a2 = list(mon2[:n])
        b2 = list(mon2[n:2*n])
        e2 = list(mon2[2*n:])
        
        # Step 1: Move g^{e1} past x^{a2} and dx^{b2}
        # g_i * x_i = q * x_i * g_i, so g_i^{e1_i} * x_i^{a2_i} = q^{e1_i * a2_i} * x_i^{a2_i} * g_i^{e1_i}
        # g_i * dx_i = q^{-1} * dx_i * g_i, so g_i^{e1_i} * dx_i^{b2_i} = q^{-e1_i * b2_i} * ...
        
        q_power = sum(e1[i] * (a2[i] - b2[i]) for i in range(n))
        coef = expand(coef * self.q**q_power)
        
        # Combine g exponents
        new_e = [e1[i] + e2[i] for i in range(n)]
        
        # Step 2: Reduce dx^{b1} * x^{a2}
        return self._reduce_dx_x(a1, b1, a2, b2, new_e, coef)

    def _reduce_dx_x(self, a1, b1, a2, b2, e, coef):
        """
        Reduce dx^{b1} * x^{a2} to PBW form.
        
        Uses the direct reduction formula:
        dx_i * x_i^a = (x_i^{a-1} / (q - q^{-1})) * (q^a * g_i - q^{-a} * g_i^{-1})
        """
        n = self.n
        
        # Use a stack instead of recursion
        stack = [(list(a1), list(b1), list(a2), list(b2), list(e), coef)]
        result = HayashiElement(self.algebra, {})
        
        while stack:
            a1_cur, b1_cur, a2_cur, b2_cur, e_cur, coef_cur = stack.pop()
            reduced = False
            
            for i in range(n):
                if b1_cur[i] > 0 and a2_cur[i] > 0:
                    a = a2_cur[i]  # current x_i exponent
                    
                    # dx_i * x_i^a = (x_i^{a-1} / (q - q^{-1})) * (q^a * g_i - q^{-a} * g_i^{-1})
                    prefactor = 1 / (self.q - self.q**(-1))
                    
                    b1_new = b1_cur.copy()
                    a2_new = a2_cur.copy()
                    b1_new[i] -= 1  # consume one dx_i
                    a2_new[i] -= 1  # reduce x_i exponent by 1
                    
                    # Term 1: q^a * g_i term
                    e_new1 = e_cur.copy()
                    e_new1[i] += 1
                    coef1 = expand(coef_cur * prefactor * self.q**a)
                    stack.append((a1_cur.copy(), b1_new, a2_new, b2_cur.copy(), e_new1, coef1))
                    
                    # Term 2: -q^{-a} * g_i^{-1} term
                    e_new2 = e_cur.copy()
                    e_new2[i] -= 1
                    coef2 = expand(coef_cur * prefactor * (-self.q**(-a)))
                    stack.append((a1_cur.copy(), b1_new, a2_new, b2_cur.copy(), e_new2, coef2))
                    
                    reduced = True
                    break
            
            if not reduced:
                # No more reductions needed, combine exponents
                a_final = [a1_cur[i] + a2_cur[i] for i in range(n)]
                b_final = [b1_cur[i] + b2_cur[i] for i in range(n)]
                final_mon = tuple(a_final + b_final + e_cur)
                result = result + HayashiElement(self.algebra, {final_mon: coef_cur})
        
        return result

    def __repr__(self):
        if not self.terms:
            return "0"
        
        parts = []
        for mon in sorted(self.terms.keys(), reverse=True):
            coef = self.terms[mon]
            mon_str = self._monomial_to_str(mon)
            
            if coef == 1:
                parts.append(mon_str if mon_str != "1" else "1")
            elif coef == -1:
                parts.append(f"-{mon_str}" if mon_str != "1" else "-1")
            else:
                coef_str = str(coef)
                if mon_str == "1":
                    parts.append(coef_str)
                else:
                    if '+' in coef_str or (coef_str.count('-') > 1) or \
                       (coef_str.count('-') == 1 and not coef_str.startswith('-')):
                        parts.append(f"({coef_str})*{mon_str}")
                    else:
                        parts.append(f"{coef_str}*{mon_str}")
        
        result = " + ".join(parts)
        result = result.replace(" + -", " - ")
        return result
    
    def _monomial_to_str(self, mon):
        n = self.n
        a = mon[:n]
        b = mon[n:2*n]
        e = mon[2*n:]
        
        parts = []
        for i in range(n):
            name = self.algebra.x_names[i]
            if a[i] == 1:
                parts.append(name)
            elif a[i] > 1:
                parts.append(f"{name}^{a[i]}")
        
        for i in range(n):
            name = self.algebra.dx_names[i]
            if b[i] == 1:
                parts.append(name)
            elif b[i] > 1:
                parts.append(f"{name}^{b[i]}")
        
        for i in range(n):
            name = self.algebra.g_names[i]
            if e[i] == 1:
                parts.append(name)
            elif e[i] == -1:
                parts.append(f"{name}^-1")
            elif e[i] != 0:
                parts.append(f"{name}^{e[i]}")
        
        return "*".join(parts) if parts else "1"
