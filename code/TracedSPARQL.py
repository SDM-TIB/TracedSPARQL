__author__ = 'Philipp D. Rohde'

import copy
import getopt
import logging
import os
import sys
import time
from multiprocessing import Process, Queue, active_children
from threading import Thread

from DeTrusty import Decomposer as DeTrustyDecomposer
from DeTrusty import Planner as DeTrustyPlanner
from DeTrusty.Decomposer import utils as utils
from DeTrusty.Decomposer.Planner import IndependentOperator as DeTrustyIndependent
from DeTrusty.Decomposer.Planner import TreePlan
from DeTrusty.Decomposer.Tree import Leaf, Node
from DeTrusty.Molecule.MTManager import ConfigFile
from DeTrusty.Operators.AnapsidOperators.Xbind import Xbind
from DeTrusty.Operators.AnapsidOperators.Xfilter import Xfilter
from DeTrusty.Operators.AnapsidOperators.Xgjoin import Xgjoin
from DeTrusty.Operators.AnapsidOperators.Xvalues import Xvalues
from DeTrusty.Sparql.Parser.services import Bind, Filter, Values, UnionBlock, JoinBlock, Optional, Triple
from DeTrusty.Sparql.Parser.services import Service as DeTrustyService
from DeTrusty.Wrapper.RDFWrapper import contact_source as contact_rdf_source

JOIN_TO_USE = Xgjoin
SHACL_SCHEMA_OVERLAP_THRESHOLD = 0.0

logging.getLogger('DeTrusty.Wrapper.RDFWrapper').setLevel(logging.ERROR)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
logger.addHandler(logging.StreamHandler())


class Mapping:
    """
    This class is used to convert between the internal representations of URIs and variables
    in DeTrusty and the API used for the constraint validation.
    """

    def __init__(self, tuple_):
        self.bindings = tuple_[0]
        self.sn = tuple_[1]
        self.exp = tuple_[2]

        # Sets to check for duplicates
        self.exp_set = {val_res[0] for val_res in self.exp}
        self.sn_set = {str(tuple_) for tuple_ in self.sn}

    @property
    def val_report(self):
        return [self.sn, self.exp]

    def __str__(self) -> str:
        return str(self.bindings) + str(self.exp) + str(self.sn)

    def add_var_prefix(self, var):
        """
        Converts the input into a variable starting with question mark ('?').
        For example, the variable 'x' is returned as '?x'.
        Returns the internal representation of a variable.
        """
        var = str(var)
        if var.startswith('?'):
            return var
        else:
            return '?' + var

    def del_var_prefix(self, var):
        """
        Converts the input into a variable that does not start with question mark ('?').
        For example, the variable '?x' is returned as 'x'.
        Returns the external representation of a variable.
        """
        var = str(var)
        if var.startswith('?'):
            return var[1:]
        else:
            return var

    def decapsulate(self, value):
        """
        Removes leading and trailing brackets ('<' and '>') from the given input.
        For example, '<https://example.org/vocab#Person>' is returned as 'https://example.org/vocab#Person'.
        Returns the external representation of a URI.
        """
        if value.startswith('<') and value.endswith('>'):
            return value[1:-1]
        else:
            return value

    def encapsulate(self, value):
        """
        Adds leading and trailing brackets ('<' and '>') to the given input.
        For example, 'https://example.org/vocab#Person' is returned as '<https://example.org/vocab#Person>'.
        Returns the internal representation of a URI.
        """
        if value.startswith('http'):
            return '<' + value + '>'
        else:
            return value

    '''
    The following methods are emulating a dictionary type such that instances of Mapping
    can be used similar to dictionaries. The goal is to make as few changes as possible to the
    operators and at the same time manage the reasoning (sn) and validation results (exp) properly.
    '''

    def __getitem__(self, var):
        var = self.add_var_prefix(var)
        value = self.bindings[var]
        return self.decapsulate(value)

    def get(self, var, default):
        var = self.add_var_prefix(var)
        if var in self.bindings:
            return self.decapsulate(self[var])
        else:
            return default

    def __setitem__(self, var, value):
        var = self.add_var_prefix(var)
        self.bindings[var] = self.encapsulate(value)
        return self

    def __delitem__(self, var):
        var = self.add_var_prefix(var)
        del self.bindings[var]
        return self

    def update(self, mapping2):
        if isinstance(mapping2, Mapping):
            self.bindings.update(mapping2.bindings)

            # Check for duplicates
            for tuple_ in mapping2.sn:
                if str(tuple_) not in self.sn_set:
                    self.sn_set.add(str(tuple_))
                    self.sn.append(tuple_)

            for r in mapping2.exp:
                if r[0] not in self.exp_set:
                    self.exp_set.add(r[0])
                    self.exp.append(r)
        elif isinstance(mapping2, dict):
            self.bindings.update({self.add_var_prefix(key): self.encapsulate(value) for key, value in mapping2.items()})
        else:
            raise Exception('Unexpected type to update a Mapping Object: ' + str(type(mapping2)))

    def __contains__(self, var):
        var = self.add_var_prefix(var)
        return var in self.bindings.keys()

    def keys(self):
        for key in self.bindings.keys():
            yield self.del_var_prefix(key)

    def values(self):
        for value in self.bindings.values():
            yield self.decapsulate(value)

    def items(self):
        for var, value in self.bindings.items():
            yield self.del_var_prefix(var), self.decapsulate(value)

    def copy(self):
        return Mapping(copy.deepcopy([self.bindings, self.sn, self.exp]))

    '''
    Operator specific functions
    '''

    def project(self, vars_):
        vars_ = [self.add_var_prefix(str(var)) for var in vars_]
        return Mapping([{key: value for key, value in self.bindings.items() if key in vars_}, self.sn, self.exp])


class TracedSPARQLDecomposer(DeTrustyDecomposer):
    """
    An extension of DeTrustyDecomposer that considers the additional configuration for the validation.
    Since the validation depends on the star-shaped decomposition, decompType is fixed to 'STAR'.
    """

    def __init__(self, query, config, joinstarslocally=True, val_config=None):
        """
        Initialize the decomposer.

        :param query: The SPARQL query to be executed
        :param config: DeTrusty's configuration, i.e., the source descriptions
        :param joinstarslocally: Indicates whether joins should be performed at the query engine, default: True
        :param val_config: Settings for the validation of the SHACL constraints
        """
        super().__init__(query=query, config=config, decompType='STAR', joinstarslocally=joinstarslocally)
        self.val_config = val_config

    def decompose(self):
        """
        Decomposes the SPARQL query and returns the decomposed SPARQL following DeTrusty's internal structure.
        """
        if not self.query:
            return None

        self.query.body = self.decomposeUnionBlock(self.query.body) if not self.query.service else self.query.body

        if not self.query.body:
            return None

        proj_vars = []
        for arg in self.query.args:
            proj_vars.extend(arg.getVars())
        proj_vars = set(proj_vars) - {'*'}  # remove * to allow COUNT(*)
        body_vars = set(self.query.body.getVars())
        if proj_vars - body_vars:
            raise Exception('The following variables have been defined in the SELECT clause but not in the body: '
                            + str(proj_vars - body_vars))

        order_by_vars = []
        for arg in self.query.order_by:
            order_by_vars.extend(arg.getVars())
        if set(order_by_vars) - proj_vars:
            raise Exception('The following variables have been defined in the ORDER BY clause but are not projected: '
                            + str(set(order_by_vars) - proj_vars))

        group_by_vars = []
        for arg in self.query.group_by:
            group_by_vars.extend(arg.getVars())
        if set(group_by_vars) - body_vars:
            raise Exception('The following variables have been defined in the GROUP BY clause but not in the body: '
                            + str(set(group_by_vars) - body_vars))

        self.query.body = self.makePlanQuery(self.query)

        return self.query

    def decomposeUnionBlock(self, ub):
        r = []
        for jb in ub.triples:
            pjb = self.decomposeJoinBlock(jb)
            if pjb:
                r.append(pjb)
        if r:
            return UnionBlock(r)
        else:
            return None

    def decomposeJoinBlock(self, jb):
        tl = []
        sl = []
        fl = []
        for bgp in jb.triples:
            if isinstance(bgp, Triple):
                tl.append(bgp)
                self.alltriplepatterns.append(bgp)
            elif isinstance(bgp, Filter):
                fl.append(bgp)
            elif isinstance(bgp, Values):
                fl.append(bgp)
            elif isinstance(bgp, Bind):
                fl.append(bgp)
            elif isinstance(bgp, Optional):
                ubb = self.decomposeUnionBlock(bgp.bgg)
                skipp = False
                if ubb is not None:
                    for ot in ubb.triples:
                        if isinstance(ot, JoinBlock) and len(ot.triples) > 1 and len(ot.filters) > 0:
                            skipp = True
                            break
                    if not skipp:
                        sl.append(Optional(ubb))
            elif isinstance(bgp, UnionBlock):
                pub = self.decomposeUnionBlock(bgp)
                if pub:
                    sl.append(pub)
            elif isinstance(bgp, JoinBlock):
                pub = self.decomposeJoinBlock(bgp)
                if pub:
                    sl.append(pub)
        if tl:
            gs = self.decomposeBGP(tl)
            if gs:
                gs.extend(sl)
                sl = gs
            else:
                return None

        fl1 = self.includeFilter(sl, fl)
        fl = list(set(fl) - set(fl1))
        if sl:
            if len(sl) == 1 and isinstance(sl[0], UnionBlock) and fl != []:
                sl[0] = self.updateFilters(sl[0], fl)
            j = JoinBlock(sl, filters=fl)
            return j
        else:
            return None

    def decomposeBGP(self, tl):
        from shaclapi.api import overlap_reduced_schemas
        stars = self.getQueryStar(tl)
        star_shape_map = self.map_star_to_shape(tl, stars)

        selectedmolecules = {}
        varpreds = {}
        starpreds = {}
        conn = self.getStarsConnections(stars)
        splitedstars = {}

        for s in stars.copy():
            ltr = stars[s]
            preds = [utils.getUri(tr.predicate, self.prefixes)[1:-1] for tr in ltr if tr.predicate.constant]
            starpreds[s] = preds
            typemols, error = self.checkRDFTypeStatemnt(ltr)
            if error:
                logger.error('No molecules found for sub-query: ' + str(ltr))
                return []
            if len(typemols) > 0:
                selectedmolecules[s] = typemols
                for m in typemols:
                    properties = [p['predicate'] for p in self.config.metadata[m]['predicates']]
                    pinter = set(properties).intersection(preds)
                    if len(pinter) != len(preds):
                        logger.error('Subquery: ' + str(stars[s]) +
                                     '\nCannot be executed, because it contains properties that ' +
                                     'do not exist in this federation.')
                        return []
                continue

            if len(preds) == 0:
                found = False
                for v in conn.values():
                    if s in v:
                        mols = [m for m in self.config.metadata]
                        found = True
                if not found:
                    varpreds[s] = ltr
                    continue
            else:
                mols = self.config.findbypreds(preds)

            if len(mols) > 0:
                if s in selectedmolecules:
                    selectedmolecules[s].extend(mols)
                else:
                    selectedmolecules[s] = mols
            else:
                splitstars = self.config.find_preds_per_mt(preds)
                if len(splitstars) == 0:
                    logger.error('Cannot find any matching molecules for:', tl)
                    return []
                else:
                    splitedstars[s] = [stars[s], splitstars, preds]
                    for m in list(splitstars.keys()):
                        selectedmolecules[str(s+'_'+m)] = [m]

        if len(varpreds) > 0:
            mols = [m for m in self.config.metadata]
            for s in varpreds:
                selectedmolecules[s] = mols

        molConn = self.getMTsConnection(selectedmolecules)
        if len(splitedstars) > 0:
            for s in splitedstars:
                newstarpreds = {utils.getUri(tr.predicate, self.prefixes)[1:-1]: tr for tr in stars[s] if tr.predicate.constant}

                for m in splitedstars[s][1]:
                    stars[str(s + '_' + m)] = [newstarpreds[p] for p in splitedstars[s][1][m]]
                    starpreds[str(s + '_' + m)] = splitedstars[s][1][m]
                del stars[s]
                del starpreds[s]

        conn = self.getStarsConnections(stars)
        res = self.pruneMTs(conn, molConn, selectedmolecules, stars)
        qpl0 = []
        qpl1 = []
        for s in res:
            if len(res[s]) == 1:
                if len(self.config.metadata[res[s][0]]['wrappers']) == 1:
                    # -- one molecule and only one wrapper --
                    endpoint = self.config.metadata[res[s][0]]['wrappers'][0]['url']
                    target_shape = self.config.metadata[res[s][0]]['target_shape'] if 'target_shape' in self.config.metadata[res[s][0]] else None
                    qpl0.append(Service('<' + endpoint + '>', list(set(stars[s])), target_shape=target_shape))
                else:
                    sources = [w['url'] for w in self.config.metadata[res[s][0]]['wrappers']
                               if len(starpreds[s]) == len(list(set(starpreds[s]).intersection(set(w['predicates']))))]
                    if len(sources) == 1:
                        # -- one molecule, several wrappers, but only one valid source --
                        endpoint = sources[0]
                        qpl0.append(Service('<' + endpoint + '>', list(set(stars[s]))))
                    elif len(sources) > 1:
                        # -- one molecule, several wrappers, and several valid sources --
                        elems = [JoinBlock([Service('<' + ep + '>', list(set(stars[s])))]) for ep in sources]
                        ub = UnionBlock(elems)
                        qpl1.append(ub)
                    else:
                        # split and join
                        wpreds = {}

                        ptrs = {utils.getUri(tr.predicate, self.prefixes)[1:-1]: tr for tr in stars[s] if tr.predicate.constant}
                        for w in self.config.metadata[res[s][0]]['wrappers']:
                            wps = [p for p in w['predicates'] if p in starpreds[s]]
                            wpreds[w['url']] = wps

                        inall = []
                        difs = {}
                        for e in wpreds:
                            if len(inall) == 0:
                                inall = wpreds[e]
                            else:
                                inall = list(set(inall).intersection(wpreds[e]))

                            if e not in difs:
                                difs[e] = wpreds[e]
                            for d in difs:
                                if e == d:
                                    continue
                                dd = list(set(difs[d]).difference(wpreds[e]))
                                if len(dd) > 0:
                                    difs[d] = dd

                                dd = list(set(difs[e]).difference(wpreds[d]))
                                if len(dd) > 0:
                                    difs[e] = dd

                        oneone = {}
                        for e1 in wpreds:
                            for e2 in wpreds:
                                if e1 == e2 or e2 + '|-|' + e1 in oneone:
                                    continue
                                pp = set(wpreds[e1]).intersection(wpreds[e2])
                                pp = list(set(pp).difference(inall))
                                if len(pp) > 0:
                                    oneone[e1 + '|-|' + e2] = pp
                        onv = []
                        [onv.extend(d) for d in list(oneone.values())]
                        difv = []
                        [difv.extend(d) for d in list(difs.values())]
                        for o in onv:
                            if o in difv:
                                toremov = []
                                for d in difs:
                                    if o in difs[d]:
                                        difs[d].remove(o)
                                        difv.remove(o)
                                    if len(difs[d]) == 0:
                                        toremov.append(d)
                                for d in toremov:
                                    del difs[d]

                        ddd = onv + difv
                        rdftype = []
                        if len(set(inall + ddd)) == len(starpreds[s]):
                            if len(inall) > 0:
                                if len(inall) == 1 and inall[0] == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                                    rdftype.extend(list(wpreds.keys()))
                                    pass
                                else:
                                    # -- one molecule, several sources, triple patterns shared by all sources --
                                    trps = [ptrs[p] for p in inall]
                                    elems = [JoinBlock([Service('<' + ep + '>', list(set(trps)))]) for ep in list(wpreds.keys())]
                                    ub = UnionBlock(elems)
                                    qpl1.append(ub)
                            if len(oneone) > 0:
                                for ee in oneone:
                                    e1, e2 = ee.split('|-|')
                                    pp = oneone[ee]
                                    if len(pp) == 1 and pp[0] == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                                        rdftype.extend([e1, e2])
                                        pass
                                    else:
                                        # -- one molecule, several sources, pairwise intersecting triple patterns --
                                        trps = [ptrs[p] for p in pp]
                                        elems = [JoinBlock([Service('<' + e1 + '>', list(set(trps)))]),
                                                 JoinBlock([Service('<' + e2 + '>', list(set(trps)))])]
                                        ub = UnionBlock(elems)
                                        qpl1.append(ub)
                            if len(difs) > 0:
                                for d in difs:
                                    # -- one molecule, several sources, triple patterns only one source --
                                    trps = [ptrs[p] for p in difs[d]]
                                    if d in rdftype:
                                        trps.append(ptrs['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'])
                                    qpl0.append(Service('<' + d + '>', list(set(trps))))
                        else:
                            return []
            else:
                md = self.metawrapperdecomposer(res[s], stars[s])
                if isinstance(md, DeTrustyService):
                    qpl0.append(md)
                else:
                    for m in md:
                        if isinstance(m, DeTrustyService):
                            qpl0.append(m)
                        else:
                            qpl1.append(m)

        if qpl0 and not self.joinlocally:  # -- join molecules in same source; if joinstarslocally=False --
            joins = {}
            g = 0
            merged = []
            for i in range(len(qpl0)):
                if i+1 < len(qpl0):
                    for j in range(i+1, len(qpl0)):
                        s = qpl0[i]
                        k = qpl0[j]
                        if s.endpoint == k.endpoint:
                            if self.shareAtLeastOneVar(k.triples, s.triples):
                                centers_s = self.get_centers(s.triples)
                                centers_k = self.get_centers(k.triples)
                                target_shapes_s = self.get_target_shapes(centers_s, star_shape_map)
                                target_shapes_k = self.get_target_shapes(centers_k, star_shape_map)
                                if not overlap_reduced_schemas(self.val_config, target_shapes_s, target_shapes_k) > SHACL_SCHEMA_OVERLAP_THRESHOLD:
                                    continue  # do not merge the stars if the overlap is not big enough
                                centers_all = list(set(centers_s).union(set(centers_k)))
                                target_shapes = self.get_var_shape_map(centers_all, star_shape_map)
                                if s.endpoint in joins:
                                    joins[s.endpoint]['target_shape'].update(target_shapes)
                                    joins[s.endpoint]['triples'].extend(s.triples + k.triples)
                                else:
                                    joins[s.endpoint] = {
                                        'triples': s.triples + k.triples,
                                        'target_shape': target_shapes
                                    }
                                merged.append(s)
                                merged.append(k)
                                joins[s.endpoint]['triples'] = list(set(joins[s.endpoint]['triples']))

            [qpl0.remove(r) for r in set(merged)]
            for s in qpl0:
                centers_s = self.get_centers(s.triples)
                if s.endpoint in joins:
                    centers_k = self.get_centers(joins[s.endpoint]['triples'])
                    target_shapes_s = self.get_target_shapes(centers_s, star_shape_map)
                    target_shapes_k = self.get_target_shapes(centers_k, star_shape_map)
                    if self.shareAtLeastOneVar(joins[s.endpoint]['triples'], s.triples) and overlap_reduced_schemas(self.val_config, target_shapes_s, target_shapes_k) > SHACL_SCHEMA_OVERLAP_THRESHOLD:
                        centers_all = list(set(centers_s).union(set(centers_k)))
                        target_shapes = self.get_var_shape_map(centers_all, star_shape_map)
                        joins[s.endpoint]['triples'].extend(s.triples)
                        joins[s.endpoint]['target_shape'] = target_shapes
                    else:
                        joins[s.endpoint + '|' + str(g)] = {
                            'triples': s.triples,
                            'target_shape': self.get_var_shape_map(centers_s, star_shape_map)
                        }
                        g += 1
                else:
                    joins[s.endpoint] = {
                        'triples': s.triples,
                        'target_shape': self.get_var_shape_map(centers_s, star_shape_map)
                    }

                joins[s.endpoint]['triples'] = list(set(joins[s.endpoint]['triples']))

            qpl0 = []
            for e in joins:
                endp = e.split('|')[0]
                qpl0.append(Service('<' + endp + '>', joins[e]['triples'], target_shape=joins[e]['target_shape']))

        if qpl0 and qpl1:
            qpl1.insert(0, qpl0)
            return qpl1
        elif qpl0 and not qpl1:
            return qpl0
        else:
            return qpl1

    def map_star_to_shape(self, tl, stars):
        """
        Maps a star-shaped sub-query to its target shape, i.e., SHACL constraints.

        :param tl: Set of triples that need to be mapped to a shape
        :param stars: The stars occurring in the SPARQL query being decomposed
        """
        selected_molecules = {}
        var_preds = {}
        star_preds = {}
        conn = self.getStarsConnections(stars)
        splitted_stars = {}

        for s in stars.copy():
            ltr = stars[s]
            preds = [utils.getUri(tr.predicate, self.prefixes)[1:-1] for tr in ltr if tr.predicate.constant]
            star_preds[s] = preds
            type_mols, error = self.checkRDFTypeStatemnt(ltr)
            if error:
                logger.error('No molecules found for sub-query: ' + str(ltr))
                return []
            if len(type_mols) > 0:
                selected_molecules[s] = type_mols
                for m in type_mols:
                    properties = [p['predicate'] for p in self.config.metadata[m]['predicates']]
                    pinter = set(properties).intersection(preds)
                    if len(pinter) != len(preds):
                        logger.error('Subquery: ' + str(stars[s]) +
                                     '\nCannot be executed, because it contains properties that ' +
                                     'do not exist in this federation.')
                        return []
                continue

            if len(preds) == 0:
                found = False
                for v in conn.values():
                    if s in v:
                        mols = [m for m in self.config.metadata]
                        found = True
                if not found:
                    var_preds[s] = ltr
                    continue
            else:
                mols = self.config.findbypreds(preds)

            if len(mols) > 0:
                if s in selected_molecules:
                    selected_molecules[s].extend(mols)
                else:
                    selected_molecules[s] = mols
            else:
                split_stars = self.config.find_preds_per_mt(preds)
                if len(split_stars) == 0:
                    logger.error('Cannot find any matching molecules for:', tl)
                    return []
                else:
                    splitted_stars[s] = [stars[s], split_stars, preds]
                    for m in list(split_stars.keys()):
                        selected_molecules[str(s + '_' + m)] = [m]

        if len(var_preds) > 0:
            mols = [m for m in self.config.metadata]
            for s in var_preds:
                selected_molecules[s] = mols

        molConn = self.getMTsConnection(selected_molecules)
        if len(splitted_stars) > 0:
            for s in splitted_stars:
                new_star_preds = {utils.getUri(tr.predicate, self.prefixes)[1:-1]: tr for tr in stars[s] if
                                  tr.predicate.constant}

                for m in splitted_stars[s][1]:
                    stars[str(s + '_' + m)] = [new_star_preds[p] for p in splitted_stars[s][1][m]]
                    star_preds[str(s + '_' + m)] = splitted_stars[s][1][m]
                del stars[s]
                del star_preds[s]

        conn = self.getStarsConnections(stars)
        res = self.pruneMTs(conn, molConn, selected_molecules, stars)
        star_shape_map = {}
        for s in res.keys():
            star_shape_map[s] = {
                'rdfmt': res[s],
                'target_shape': self.config.metadata[res[s][0]]['target_shape'] if 'target_shape' in self.config.metadata[res[s][0]] else None
            }
        return star_shape_map

    def get_centers(self, triples):
        """
        Gets the variable names that represent the different stars in a set of triples.
        """
        centers = set()
        for triple in triples:
            centers.add(triple.subject.name)
        return list(centers)

    def get_target_shapes(self, centers, star_shape_map):
        """
        Gets the list of target shapes for the given list of stars.
        """
        target_shapes = set()
        for center in centers:
            if center in star_shape_map.keys():
                target_shapes.add(star_shape_map[center]['target_shape'])
        return list(target_shapes)

    def get_var_shape_map(self, centers, star_shape_map):
        """
        Creates a dictionary containing the target shape for each variable in the list that represents a star.
        """
        var_shape_map = {}
        for center in centers:
            if center in star_shape_map.keys():
                var_shape_map[center] = [star_shape_map[center]['target_shape']]
        return var_shape_map


class Service(DeTrustyService):
    """
    An extension of the DeTrustyService class in order to propagate the target shape of a sub-query.
    """

    def __init__(self, endpoint, triples, limit=-1, filter_nested=None, target_shape=None):
        """
        Initializes the Service instance.

        :param endpoint: The endpoint to contact for the sub-query
        :param triples: The triples to include in the query sent to the endpoint.
        :param limit: The maximum number of triples to return per call, default -1 means no limit
        :param filter_nested: List of nested filters, default None
        :param target_shape: The target shape to include in the validation, default None
        """
        super().__init__(endpoint, triples, limit, filter_nested)
        self.target_shape = target_shape

    def instantiate(self, d):
        if isinstance(self.triples, list):
             new_triples = [t.instantiate(d) for t in self.triples]
        else:
             new_triples = self.triples.instantiate(d)
        return Service('<' + self.endpoint + '>', new_triples, self.limit, target_shape=self.target_shape)

    def instantiateFilter(self, d, filter_str):
        new_filters = []
        new_filters.extend(self.filter_nested)
        new_filters.append(filter_str)
        return Service('<' + self.endpoint + '>', self.triples, self.limit, new_filters, self.target_shape)


class TracedSPARQLPlanner(DeTrustyPlanner):
    """
    An extension of DeTrustyPlanner that considers the validation config.
    """

    def __init__(self, query, wc, contact, endpointType, config, val_config):
        """
        Initialized the TracedSPARQLPlanner instance.

        :param query: The decomposed SPARQL query for which to create the query plan
        :param wc: Indicates whether counts should be kept
        :param contact: The function to be used in order to contact the endpoints
        :param endpointType: The types of endpoints to be considered, only SPARQL endpoints are supported
        :param config: DeTrusty's configuration, i.e., the source descriptions
        :param val_config: The configuration for the validation
        """
        super().__init__(query, wc, contact, endpointType, config)
        self.val_config = val_config

    def includePhysicalOperators(self, tree):
        if isinstance(tree, Leaf):
            if isinstance(tree.service, DeTrustyService):
                val_config = self.val_config if isinstance(tree.service, Service) else None
                if tree.filters == []:
                    return IndependentOperator(self.query, tree, self.contact, self.config, val_config)
                else:
                    n = IndependentOperator(self.query, tree, self.contact, self.config, val_config)
                    for f in tree.filters:
                        vars_f = f.getVarsName()
                        if set(n.vars) & set(vars_f) == set(vars_f):
                            f.expr.replace_prefix(self.query.prefs)
                            n = TreePlan(Xfilter(f), n.vars, n)
                    return n
            elif isinstance(tree.service, UnionBlock):
                return self.includePhysicalOperatorsUnionBlock(tree.service)
            elif isinstance(tree.service, JoinBlock):
                if tree.filters == []:
                    return self.includePhysicalOperatorsJoinBlock(tree.service)
                else:
                    n = self.includePhysicalOperatorsJoinBlock(tree.service)
                    for f in tree.filters:
                        vars_f = f.getVarsName()
                        if set(n.vars) & set(vars_f) == set(vars_f):
                            f.expr.replace_prefix(self.query.prefs)
                            n = TreePlan(Xfilter(f), n.vars, n)
                    return n
            else:
                logger.error('Error: Type not considered: tree.service ' + str(type(tree.service)) + str(tree.service))

        elif isinstance(tree, Node):
            left_subtree = self.includePhysicalOperators(tree.left)
            right_subtree = self.includePhysicalOperators(tree.right)
            if tree.filters == []:
                return self.includePhysicalOperatorJoin(left_subtree, right_subtree)
            else:
                n = self.includePhysicalOperatorJoin(left_subtree, right_subtree)
                for f in tree.filters:
                    vars_f = f.getVarsName()
                    if set(n.vars) & set(vars_f) == set(vars_f):
                        if isinstance(f, Filter):
                            f.expr.replace_prefix(self.query.prefs)
                            n = TreePlan(Xfilter(f), n.vars, n)
                        elif isinstance(f, Values):
                            for dbv in f.data_block_val:
                                for arg in dbv:
                                    arg.replace_prefix(self.query.prefs)
                            n = TreePlan(Xvalues(f), n.vars, n)
                    if isinstance(f, Bind):
                        n.vars = set(n.vars) | set(vars_f)
                        if n.vars & set(vars_f) == set(vars_f):
                            f.expr.replace_prefix(self.query.prefs)
                            n = TreePlan(Xbind(f), n.vars, n)
            return n


class IndependentOperator(DeTrustyIndependent):
    """
    An extension of DeTrustyIndependent that keeps track of the target shape of the sub-query.
    """

    def __init__(self, query, tree, c, config, val_config):
        """
        Initializes the IndependentOperator instance.

        :param query: The decomposed SPARQL query
        :param tree: The sub-tree representing the independent operator, i.e., sub-query
        :param c: The function to use for contacting the endpoint
        :param config: The configuration of DeTrusty, i.e., the source descriptions
        :param val_config: The configuration for the constraint validation
        """
        super().__init__(query, tree, c, config)
        self.val_config = val_config

    def instantiate(self, d):
        new_tree = self.tree.instantiate(d)
        return IndependentOperator(self.query, new_tree, self.contact, self.config, self.val_config)

    def instantiateFilter(self, vars_instantiated, filter_str):
        new_tree = self.tree.instantiateFilter(vars_instantiated, filter_str)
        return IndependentOperator(self.query, new_tree, self.contact, self.config, self.val_config)

    def execute(self, outputqueue, processqueue=Queue()):
        """
        Executes the sub-query, i.e., creates a new process which is using the specified
        method to contact the endpoint. The results are written to a queue.

        :param outputqueue: The queue to put the query results in
        :param processqueue: The queue to manage the running processes, defaults to a new queue
        """
        if self.tree.service.limit == -1:
            self.tree.service.limit = 10000  # TODO: Fixed value, this can be learnt in the future

        # Evaluate the independent operator.
        target_shape = self.tree.service.target_shape if self.val_config else None
        p = Process(target=self.contact, args=(self.server, self.query_str, outputqueue, self.val_config, target_shape, self.config, self.tree.service.limit))
        p.start()


def contact_source(server, query, queue, val_config=None, target_shape=None, config=None, limit=-1):
    """
    Determines which function to use for contacting the SPARQL endpoint, i.e., whether to use
    the original implementation of DeTrusty or the TracedSPARQL implementation which is adding
    SHACL validation results as a trace to the SPARQL query result.

    :param server: The external SPARQL endpoint
    :param query: The star-shaped query to be executed
    :param queue: The queue to put the query results in
    :param val_config: The configuration for the validation
    :param target_shape: The name of the shape which is focused by the star-shaped query
    :param config: DeTrusty's configuration, i.e., source descriptions
    :param limit: The maximum number of triples to return per call, default -1 means no limit
    """
    try:
        if val_config is not None:
            contact_val_source(server, query, queue, val_config, target_shape, config=config, limit=limit)
        else:
            contact_rdf_source(server, query, queue, config=config, limit=limit)
    except Exception as e:
        queue.put('EOF')
        logger.error('EXCEPTION in contact source: ', str(e))
        import sys
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        emsg = repr(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.error(emsg)


def contact_val_source(server, query, queue, val_config, target_shape, config=None, limit=-1):
    """
    Contacts the validation API which gives the joined validation results by contacting the
    external SPARQL endpoint and asking the given backend for validation results.

    :param server: The external SPARQL endpoint
    :param query: The star-shaped query to be executed
    :param queue: The queue to put the query results in
    :param val_config: The configuration for the validation
    :param target_shape: The name of the shape which is focused by the star-shaped query
    :param config: DeTrusty's configuration, i.e., source descriptions
    :param limit: The maximum number of triples to return per call, default -1 means no limit
    """
    from shaclapi.api import run_multiprocessing, get_result_queue
    logging.getLogger().setLevel(logging.ERROR)

    params = {
        'external_endpoint': server,
        'query': query,
        'targetShape': target_shape,
        'config': val_config['api_config'],
        'schemaDir': val_config['schema_directory'],
        'test_identifier': val_config['test_id'],
        'output_directory': val_config['output_directory'],
        'write_stats': True
    }

    in_queue = get_result_queue()
    t_api = Thread(target=run_multiprocessing, args=(params, in_queue))
    t_api.start()
    new_result = in_queue.receiver.get()
    while new_result != 'EOF':
        queue.put(Mapping(new_result))
        new_result = in_queue.receiver.get()
    queue.put('EOF')
    result = True
    return result


def get_options():
    """
    Gets the command line arguments.
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h:q:c:r:a:s:i:p:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    query_file = None
    config_file = './Config/rdfmts.json'
    print_result = True
    api_config = None
    schema_path = None
    query_id = 'Q'
    output_dir = './'
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-q':
            query_file = arg
        elif opt == '-c':
            config_file = arg
        elif opt == '-r':
            print_result = eval(arg)
        elif opt == '-a':
            api_config = arg
        elif opt == '-s':
            schema_path = arg
        elif opt == '-i':
            query_id = arg
        elif opt == '-p':
            output_dir = arg

    if not query_file or (api_config and not schema_path):
        usage()
        sys.exit(1)

    val_config = None
    if api_config:
        val_config = {
            'api_config': api_config,
            'schema_directory': schema_path,
            'test_id': query_id,
            'output_directory': os.path.join(os.sep, 'shaclAPI', 'output')
        }

    return query_file, config_file, print_result, val_config, query_id, output_dir


def usage():
    """
    Prints the usage message for TracedSPARQL.
    """
    usage_str = 'Usage: {program} -q <query_file> -c <config_file> -o <sparql1.1> -r <print_result> ' \
                '-a <api_config> -s <schema_path> -i <query_id> -d <decomposition> -p <output_dir>' \
                '\nwhere \n' \
                '<query_file> path to the file containing the query to be executed\n' \
                '<config_file> path to the config file containing information about the federation of endpoints\n' \
                '<print_result> is one in [True, False] (default True), when False, only metadata is returned\n' \
                '<api_config> path to config file for the SHACL API; omit when no SHACL shapes are to be validated\n' \
                '<schema_path> path to the SHACL shape schema to validate the data against\n' \
                '<query_id> ID used to identify the given query in the logs\n' \
                '<output_dir> path where to store the performance metrics'
    print(usage_str.format(program=sys.argv[0]), )


def main(query_file, config_file, print_result, val_config, query_id, output_dir):
    """
    Runs a SPARQL query with TracedSPARQL.

    :param query_file: Path to the file containing the SPARQL query to be executed
    :param config_file: Path to the file containing the source descriptions
    :param print_result: Indicates whether to print the results of the SPARQL query
    :param val_config: Path to the configuration file for the constraint validation
    :param query_id: ID used to identify the query
    :param output_dir: Output path for the files containing the statistics of running TracedSPARQL
    """
    try:
        query = open(query_file, 'r', encoding='utf8').read()
        config = ConfigFile(config_file)

        trace = []
        test_name = val_config['test_id'] if val_config else query_id
        approach_name = os.path.basename(val_config['api_config']) if val_config else 'no_SHACL'

        if val_config is not None:
            import shaclapi.api  # needs to be imported; processes do not shut down properly when not imported

        start_time = time.time()
        decomposer = TracedSPARQLDecomposer(
            query,
            config,
            joinstarslocally=True if val_config is None else False,
            val_config=val_config
        )
        decomposed_query = decomposer.decompose()

        if decomposed_query is None:
            return {'results': {}, 'error': 'The query cannot be answered by the endpoints in the federation.'}

        planner = TracedSPARQLPlanner(
            query=decomposed_query,
            wc=True,
            contact=contact_source,
            endpointType='RDF',
            config=config,
            val_config=val_config
        )
        plan = planner.createPlan()

        output = Queue()
        plan.execute(output)

        result = []
        r = output.get()
        card = 0
        while r != 'EOF':
            card += 1
            trace.append(
                {'test': test_name, 'approach': approach_name, 'answer': card, 'time': time.time() - start_time}
            )
            if print_result:
                res = {}
                for key, value in r.items():
                    res[key] = value

                result.append(res)
            r = output.get()
        end_time = time.time()

        # Write metrics file
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, 'metrics.csv')
        mode = 'a' if os.path.isfile(metrics_file) else 'w'
        metrics_entry = {
            'test': test_name,
            'approach': approach_name,
            'tfft': trace[0]['time'] if len(trace) > 0 else 'NaN',
            'totaltime': trace[-1]['time'] if len(trace) > 0 else 'NaN',
            'comp': card
        }
        with open(metrics_file, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write(','.join(list(metrics_entry.keys())) + '\n')
            f.write(','.join(list(map(str, metrics_entry.values()))) + '\n')

        if val_config:
            api_stats_file = os.path.join(val_config['output_directory'], 'stats.csv')

            # The file might not have been written yet, so wait for it
            while not os.path.isfile(api_stats_file):
                time.sleep(1)

            with open(api_stats_file, 'r', encoding='utf-8') as f:
                header = f.readline()  # just consume it, we do not need it
                query_time = 0
                val_time = 0
                join_time = 0
                line = f.readline()
                while line:
                    line_array = line.split(',')
                    query_time += float(line_array[3]) if line_array[3] != 'NaN' else 0
                    val_time += float(line_array[4]) if line_array[4] != 'NaN' else 0
                    join_time += float(line_array[5]) if line_array[5] != 'NaN' else 0
                    line = f.readline()

            # Delete sub-query stats files; they are no longer needed
            if os.path.isfile(api_stats_file):
                os.remove(api_stats_file)

        # Write stats file
        stats_file = os.path.join(output_dir, 'stats.csv')
        mode = 'a' if os.path.isfile(stats_file) else 'w'
        stats_entry = {
            'test': test_name,
            'approach': approach_name,
            'total_execution_time': end_time - start_time,
            'query_time': query_time if val_config else 'NaN',
            'network_validation_time': val_time if val_config else 'NaN',
            'query_val_join_time': join_time if val_config else 'NaN'
        }
        with open(stats_file, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write(','.join(list(stats_entry.keys())) + '\n')
            f.write(','.join(list(map(str, stats_entry.values()))) + '\n')

        # sometimes, subprocesses are still running even though they are done
        # TODO: this is supposed to be a workaround, we should solve the issue at the source
        active = active_children()
        for child in active:
            child.kill()

        return {
            'head': {'vars': decomposed_query.variables()},
            'cardinality': card,
            'results': {'bindings': result} if print_result else 'printing results was disabled',
            'execution_time': end_time - start_time,
            'output_version': '2.0'
        }
    except Exception as e:
        logger.error('EXCEPTION: ' + str(e))
        exit(-1)


if __name__ == '__main__':
    query_file, config_file, print_result, val_config, query_id, output_dir = get_options()
    main(
        query_file=query_file,
        config_file=config_file,
        print_result=print_result,
        val_config=val_config,
        query_id=query_id,
        output_dir=output_dir
    )

