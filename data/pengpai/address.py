class address():
    def __init__(self, abroad=None, specific_traffic=None,
                 province=None, city=None, county=None,
                 township=None, village=None, poi=None,
                 infect=False) -> None:
        self.province = province
        self.city = city
        self.county = county
        self.township = township
        self.village = village
        self.poi = poi
        self.abroad = abroad
        self.inffect = infect
        self.specific = specific_traffic

    def province_addr(self):
        return str(self.province).strip().replace('None', '')

    def city_addr(self):
        return (self.province_addr()+str(self.city)).strip().replace('None', '')

    def county_addr(self):
        return (self.city_addr()+str(self.county)).strip().replace('None', '')

    def township_addr(self):
        return (self.county_addr()+str(self.township)).strip().replace('None', '')

    def village_addr(self):
        return (self.township_addr()+str(self.village)).strip().replace('None', '')

    def poi_addr(self):
        return (self.village_addr()+str(self.poi)).strip().replace('None', '')

    def full_addr(self):
        return (str(self.abroad)+str(self.specific)+self.poi_addr()).strip().replace('None','')
    
    def get_addr(self,level):
        addrmap = {
            'province':self.province_addr(),
            'city':self.city_addr(),
            'county':self.county_addr(),
            'township':self.township_addr(),
            'village':self.village_addr(),
            'poi':self.poi_addr(),
            'full':self.full_addr()
        }
        return addrmap[level]
    def get_single_addr(self,level):
        single_addr_map = {
            'province':self.province,
            'city':self.city,
            'county':self.county,
            'township':self.township,
            'village':self.village,
            'poi':self.poi,
            'full':self.full_addr()
        }
        return single_addr_map[level]