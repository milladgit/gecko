//
// Created by millad on 11/28/18.
//

#ifndef __GECKO_GECKOCONFIGLOADER_H
#define __GECKO_GECKOCONFIGLOADER_H

#include "geckoDataTypes.h"

GeckoError 	geckoLoadConfigWithFile(char *filename);
GeckoError 	geckoLoadConfigWithEnv();


#endif //__GECKO_GECKOCONFIGLOADER_H
