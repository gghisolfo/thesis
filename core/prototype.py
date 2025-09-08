

class Prototype:

    def __init__(self, property_variance_dict, rules, same_shape):

        self.property_variance_dict = {prop_class: variance for prop_class, variance in property_variance_dict.items()}

        self.rules = rules[:]

        self.same_shape = same_shape

    def test(self, obj, ind_id= -1):

        score = 0

        for prop_class in obj.properties[obj.frames_id[-1]].keys():
            variance = False
            for i_fid in range(len(obj.frames_id) - 1):
                if obj.properties[obj.frames_id[i_fid]][prop_class] != obj.properties[obj.frames_id[i_fid + 1]][prop_class]:
                    variance = True
                    break
            if prop_class in self.property_variance_dict.keys():
                if self.property_variance_dict[prop_class] != variance:
                    if variance:
                        return False, -100
                    else:
                        score -= 1
                elif variance:
                    score += 1
            else:
                #print(self.property_variance_dict)
                #print(obj.sequence[0].description)
                #print(f'{prop_class.__name__} not in self.property_variance_dict')
                return False, -100
            
        #return True, score

        if ind_id == 0 and 'wall_left' in obj.sequence[0].description: debug= False
        else: debug= False

        if debug:
            print('================================================================================================================================')
            print(f'{obj.sequence[0].description} in ind_{ind_id}')
            print(f'against prototype {self.property_variance_dict}')
            print('================================================================================================================================')

        for cf in obj.frames_id:
            if debug: print(f'cf: {cf}')
            for rule in self.rules:
                if debug: print(f'\n=================\nchecking rule: {rule.my_hash()}')
                triggered, effects, cause_offset = rule.trigger(obj, cf, debug= debug)
                if debug: print(f'triggered: {triggered} - effects: {effects} - cause_offset: {cause_offset}')
                if triggered:
                    ef = cf + cause_offset
                    if debug:
                        print(f'ef: {ef}')
                        print(f'unexplaineds: {obj.unexplained}')
                    if ef in obj.unexplained.keys():
                        if debug:
                            print(f'result: {all(any(effect.test(unexpl) for unexpl in obj.unexplained[ef]) for effect in effects)}')
                        if all(any(effect.test(unexpl) for unexpl in obj.unexplained[ef]) for effect in effects):
                            score += 10
                        else:
                            return False, -100
                    else:
                        return False, -100
        
        return True, score